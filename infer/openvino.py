import os
import time
import glob
import numpy as np
import torch
import yaml
import sys
from pathlib import Path
import logging as log
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from skimage.io import imsave as save_rgb
from codecarbon import track_emissions

import openvino.runtime as ov
from openvino.runtime import Core, get_version, PartialShape

from tifffile import imread, imsave
from codecarbon import EmissionsTracker
from torchmetrics import Dice, StructuralSimilarityIndexMeasure, PearsonCorrCoef
from tqdm.contrib import tenumerate

from cellpose import core, utils, io, models, metrics
from monai.inferers import sliding_window_inference

from utils import Dict2ObjParser,AverageMeter,timer
from parse_info import Mmv_im2imParser, OmniposeParser

from mmv_im2im.configs.config_base import (
    ProgramConfig,
    parse_adaptor,
    configuration_validation,
)
from mmv_im2im.data_modules import get_data_module
from mmv_im2im.utils.misc import generate_test_dataset_dict, parse_config
from mmv_im2im.utils.for_transform import parse_monai_ops_vanilla

def create_model(opv_path):
    core = Core()
    config = {"PERFORMANCE_HINT": "THROUGHPUT"}
    model = core.compile_model(opv_path,'CPU',config)
    '''
    model: <class 'openvino.runtime.ie_api.CompiledModel'>
    model.inputs: List(<class 'openvino.pyopenvino.ConstOutput'>) 
        - get_shape()
        - get_any_name()
    '''
    return model


class OpenVINOModel(object): 
    def __init__(self, model,config_yml):
        configure = Dict2ObjParser(config_yml).parse()
        model_name = configure.model.model_name
        self._base_model = model
        self._nets = {}
        self._model_id = "default"
        self.infer_path = config_yml['model'][model_name]['model_path']
        self.input_names = configure.data.io.input_names
        self.output_names = configure.data.io.output_names
        self.exec_net = self._init_model()

    def _init_model(self):
        if self._model_id in self._nets:
            return self._nets[self._model_id]
                # Load a new instance of the model with updated weights
        if self._model_id != "default":
            self._base_model.load_model(self._model_id, device=None)
        self.opv_model = create_model(self.infer_path)
        infer_request = self.opv_model.create_infer_request()
        self._nets[self._model_id] = infer_request
        return infer_request

    def __call__(self, inp):
        batch_size = inp.shape[0]
        if batch_size > 1:
            output = {key:[] for key in self.output_names}
            for i in range(batch_size):
                outs = self.exec_net.infer({self.input_names[0]: inp[i : i + 1]})
                outs = {out.get_any_name(): value for out, value in outs.items()}
                for key,value in outs.items():
                    output[key].append(value)
            output = {key:torch.tensor(np.concatenate(value)) for key,value in output.items()}
            return list(output.values()) if len(output.values())>1 else list(output.values())[0]
        else:
            outs = self.exec_net.infer({self.input_names[0]: inp})
            outs = {out.get_any_name(): value for out, value in outs.items()}
            outs = {key: torch.tensor(value) for key,value in outs.items()}
            return list(outs.values()) if len(outs.values())>1 else list(outs.values())[0]

    def eval(self):
        pass
    
    def load_model(self, path, device):
        self._model_id = path
        return self

class OmniposeInfer():
    
    def __init__(self, config_yml) -> None: #define the model
        configure = Dict2ObjParser(config_yml).parse()
        model_name = configure.model.model_name
        cfg_path = configure.model.omnipose.config_path
        self.base_path = os.path.split(cfg_path)[0]
        with open(cfg_path, "r") as stream:
            cfg_yml = yaml.safe_load(stream)
            self.cfg = Dict2ObjParser(cfg_yml).parse()
        infer_path = configure.model.omnipose.model_path
        self.parser = OmniposeParser(configure)
        model = self.parser.parse_model()
        # model.mkldnn = False
        # model.net.mkldnn = False
        model.net = OpenVINOModel(model.net,config_yml)
        self.model = model
        # net = OpenVINOModel(model.net,config_yml)
        # self.model = net
        self.data_dir = self.cfg.data_path
        self.input_size = configure.data.input_size
        self.device = torch.device("cuda" if self.cfg.use_gpu else "cpu")
        
        
    def prepare_data(self):
        self.files = io.get_image_files(os.path.join(self.data_dir,'im'), '_masks')
        mask_files = io.get_image_files(os.path.join(self.data_dir,'gt'), '_masks')
        images = [io.imread(f) for f in self.files] 
        # just for in house data:
        
        self.images = [img[:,:,0] for img in images]
        self.test_masks = [io.imread(f).astype(np.uint16) for f in mask_files]
        
        
    def save_result(self):
        io.save_masks(self.images, 
              self.masks, 
              self.flows, 
              self.files, 
              savedir = self.base_path,
              save_txt=False, # save txt outlines for ImageJ
              save_flows=False, # save flows as TIFFs
              )
        threshold = [0.5, 0.75, 0.9]
        ap,tp,fp,fn = metrics.average_precision(self.test_masks, self.masks, threshold=threshold)    
        print(ap)
    
    def calculate_infer_time(self,num):
        infer_time = AverageMeter()
        infer_data = [torch.randn(1,*self.input_size,device = self.device) for _ in range(num)]
        for x in infer_data:
            end = time.time()
            y_hat = self.model.net(x)
            infer_time.update(time.time()-end)
        avg_infer_time = infer_time.avg
        print(f"average inference time is {avg_infer_time:.3f}")
    
    def calculate_energy(self,num): # the cpu/gpu energy consumed by the class.
        infer_data = [torch.randn(1,*self.input_size,device = self.device) for _ in range(num)]
        # self.model.net.to(self.device)
        tracker = EmissionsTracker(measure_power_secs = 1,
                                   output_dir = self.base_path)                             
        tracker.start()
        with torch.no_grad():
            for x in infer_data:
                y_hat = self.model.net(x)
        emissions: float = tracker.stop()
        print(emissions)
    
    
    @timer
    def core_infer(self):
        masks, flows, _ = self.model.eval(self.images, 
                                  channels=self.cfg.channels,
                                  diameter=self.cfg.diameter,
                                  flow_threshold=self.cfg.flow_threshold,
                                  cellprob_threshold=self.cfg.cellprob_threshold,
                                  )
        self.masks = masks
        self.flows = flows
    
    def run_infer(self):
        self.prepare_data()
        self.core_infer()
        self.save_result()

class Mmv_im2imInfer():
    
    def __init__(self, config_yml) -> None: #define the model
        configure = Dict2ObjParser(config_yml).parse()
        model_name = configure.model.model_name
        cfg_path = configure.model.mmv_im2im.config_path
        self.base_path = os.path.split(cfg_path)[0]
        self.parser = Mmv_im2imParser(configure)
        model = self.parser.parse_model()
        net = OpenVINOModel(model.net,config_yml)
        self.model = net
        self.config = self.parser.config
        self.data_cfg = self.config.data
        self.model_cfg = self.config.model
        self.device = torch.device('cpu')
        self.input_size = configure.data.input_size
        
    def prepare_data(self):
        self.dataset_list = generate_test_dataset_dict(
            self.data_cfg.inference_input.dir, self.data_cfg.inference_input.data_type
        )

        self.dataset_length = len(self.dataset_list)
        if "Z" in self.data_cfg.inference_input.reader_params["dimension_order_out"]:
            self.spatial_dims = 3
        else:
            self.spatial_dims = 2

        if self.data_cfg.preprocess is not None:
            # load preprocessing transformation
            self.pre_process = parse_monai_ops_vanilla(self.data_cfg.preprocess)
    
    def save_result(self,pred,out_fn):
        if out_fn.suffix == ".npy":
                    np.save(out_fn, pred)
        else:
            if len(pred.shape) == 2:
                OmeTiffWriter.save(pred, out_fn, dim_order="YX")
            elif len(pred.shape) == 3:
                # 3D output, for 2D data
                if self.spatial_dims == 2:
                    # save as RGB or multi-channel 2D
                    if pred.shape[0] == 3:
                        if out_fn.suffix != ".png":
                            out_fn = out_fn.with_suffix(".png")
                        save_rgb(out_fn, np.moveaxis(pred, 0, -1))
                    else:
                        OmeTiffWriter.save(pred, out_fn, dim_order="CYX")
                elif self.spatial_dims == 3:
                    OmeTiffWriter.save(pred, out_fn, dim_order="ZYX")
                else:
                    raise ValueError("Invalid spatial dimension of pred")
            elif len(pred.shape) == 4:
                if self.spatial_dims == 3:
                    OmeTiffWriter.save(pred, out_fn, dim_order="CZYX")
                elif self.spatial_dims == 2:
                    if pred.shape[0] == 1:
                        if pred.shape[1] == 1:
                            OmeTiffWriter.save(pred[0, 0], out_fn, dim_order="YX")
                        elif pred.shape[1] == 3:
                            if out_fn.suffix != ".png":
                                out_fn = out_fn.with_suffix(".png")
                            save_rgb(
                                out_fn,
                                np.moveaxis(
                                    pred[0,],
                                    0,
                                    -1,
                                ),
                            )
                        else:
                            OmeTiffWriter.save(
                                pred[0,],
                                out_fn,
                                dim_order="CYX",
                            )
                    else:
                        raise ValueError("invalid 4D output for 2d data")
            elif len(pred.shape) == 5:
                assert pred.shape[0] == 1, "error, found non-trivial batch dimension"
                OmeTiffWriter.save(
                    pred[0,],
                    out_fn,
                    dim_order="CZYX",
                )
            else:
                raise ValueError("error in prediction output shape")
    
    def process_one_image(self,ds):
        img = AICSImage(ds).reader.get_image_dask_data(
                    **self.data_cfg.inference_input.reader_params
                )
        x = img.compute()
        x = torch.tensor(x.astype(np.float32))
        x = self.pre_process(x) #normalize to [0,1]
        x = x.as_tensor().unsqueeze(0) #bczyx
        del img
        return x
     
    def evaluate(self,pred_dir,gt_dir,pred_data_type,gt_data_type,metric):
        metric_table = dict(SSIM = StructuralSimilarityIndexMeasure(),
                        Dice = Dice(average='micro',ignore_index=0),
                        Pearson = PearsonCorrCoef())
        our_metric = {}
        for k in metric:
            try:
                our_metric[k] = metric_table[k].to(torch.device('cpu'))
            except:
                raise TypeError("metric %s is not supported" % metric) 
        #read gt/pred file in order. Suppose the file names are the same.
        gt_list = generate_test_dataset_dict(gt_dir, gt_data_type)
        pred_list = generate_test_dataset_dict(pred_dir, pred_data_type)
        if self.data_cfg.preprocess is not None:
            # load preprocessing transformation
            self.pre_process = parse_monai_ops_vanilla(self.data_cfg.preprocess)
            
        for i,(gt,pred) in tenumerate(zip(gt_list,pred_list)):
            gt = self.process_one_image(gt)
            pred = self.process_one_image(pred)
            # gt = gt.to(self.device)
            # pred = pred.to(self.device)
            if "Dice" in metric: #for semantic_seg
                act_layer = torch.nn.Softmax(dim=1)
                yhat_act = act_layer(pred).numpy()
                out_img = np.argmax(yhat_act, axis=1, keepdims=True).astype(np.uint8)
                out_img = torch.from_numpy(out_img)
                our_metric["Dice"].update(gt,out_img)
            if "SSIM" in metric: #for domain adaptation
                our_metric["SSIM"].update(gt,pred)
            if "Pearson" in metric: # for domain adaptation
                gt = torch.flatten(gt.squeeze(0).squeeze(0))
                pred = torch.flatten(pred.squeeze(0).squeeze(0))
                our_metric["Pearson"].update(gt,pred)
            
        metric_summary = {}
        for k,v in our_metric.items():
            score = v.compute()
            print(k + f" score is {score:.3f}")
            metric_summary[k] = score 
    
    def calculate_infer_time(self,num): #only a slice, not the whole image. circulate num times, take the average.
        infer_time = AverageMeter()
        infer_data = [torch.randn(1,*self.input_size) for _ in range(num)]
        for x in infer_data:
            end = time.time()
            y_hat = self.model(x)
            infer_time.update(time.time()-end)
        avg_infer_time = infer_time.avg
        print(f"average inference time is {avg_infer_time:.3f}")
    
    def calculate_energy(self,num): # the cpu/gpu energy consumed by the class.
        infer_data = [torch.randn(1,*self.input_size) for _ in range(num)]
        tracker = EmissionsTracker(measure_power_secs = 1,
                               tracking_mode = 'process',
                               output_dir = self.base_path)
        tracker.start()
        with torch.no_grad():
            for x in infer_data:
                y_hat = self.model(x)
        emissions: float = tracker.stop()
        print(emissions)
        
    def run_infer(self):
        self.prepare_data()
        use_window_inference = True
        infer_time = AverageMeter()
        with torch.no_grad():
            for i, ds in tenumerate(self.dataset_list):
                fn_core = Path(ds).stem
                suffix = self.data_cfg.inference_output.suffix
                img = AICSImage(ds).reader.get_image_dask_data(
                    **self.data_cfg.inference_input.reader_params
                )
                x = img.compute()
                x = torch.tensor(x.astype(np.float32))
                if self.pre_process is not None:
                    x = self.pre_process(x)
                x = x.unsqueeze(0).as_tensor() 
                # calcuate avg time for the whole image
                end = time.time()
                if (
                self.model_cfg.model_extra is not None
                and "sliding_window_params" in self.model_cfg.model_extra
                and use_window_inference
            ):
                    y_hat = sliding_window_inference(
                        inputs=x,
                        predictor=self.model,
                        device=torch.device("cpu"),
                        **self.model_cfg.model_extra["sliding_window_params"],
                    )
                else:
                    y_hat = self.model(x)
                infer_time.update(time.time()-end)
                pred = y_hat.squeeze(0).squeeze(0).numpy()
                out_fn = (
                    Path(self.data_cfg.inference_output.path)
                    / f"{fn_core}.tif" #need to add _{suffix} if input and output path are different.
                )
                self.save_result(pred,out_fn)
        avg_infer_time = infer_time.avg
        print(f"average inference time is {avg_infer_time:.3f}")