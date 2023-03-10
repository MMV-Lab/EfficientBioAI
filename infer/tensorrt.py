import os
import time
import glob
import numpy as np
import torch
import yaml
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from skimage.io import imsave as save_rgb
import tensorrt as trt
from tifffile import imread, imsave
from tqdm import tqdm
from codecarbon import EmissionsTracker
from torchmetrics import Dice, StructuralSimilarityIndexMeasure, PearsonCorrCoef
from tqdm.contrib import tenumerate
from codecarbon import track_emissions
import tifffile 
from cellpose import core, utils, io, models, metrics
from monai.inferers import sliding_window_inference

from utils import Dict2ObjParser,AverageMeter,timer
from parse_info import Mmv_im2imParser, OmniposeParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = trt.Logger(trt.Logger.INFO)

from mmv_im2im.configs.config_base import (
    ProgramConfig,
    parse_adaptor,
    configuration_validation,
)
from mmv_im2im.data_modules import get_data_module
from mmv_im2im.utils.misc import generate_test_dataset_dict, parse_config
from mmv_im2im.utils.for_transform import parse_monai_ops_vanilla

from typing import Any, Dict, List, Optional, Tuple, Union, Sequence


def trt_version():
    return trt.__version__

def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)
    
def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)

class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self.engine = engine
        if self.engine is not None:
            # engine创建执行context
            self.context = self.engine.create_execution_context()
            self.context.active_optimization_profile = 0

        self.input_names = input_names
        self.output_names = output_names

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            # 设定shape 
            self.context.set_binding_shape(idx, tuple(inputs[i].shape))
            bindings[idx] = inputs[i].contiguous().data_ptr()
        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream) 
        # self.context.execute_v2(bindings)                              

        outputs = tuple(outputs)
        return outputs[0] if len(outputs) == 1 else reversed(outputs)


def create_trt_model(trt_path: str) -> TRTModule:
    """create tensorrt model by reading the serialized engine.

    Args:
        trt_path (str): path of the serialized engine.

    Returns:
        TRTModule: TRTModule defined before. 
    """
    with open(trt_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine=runtime.deserialize_cuda_engine(f.read())
    input_name = []
    output_name = []
    for idx in range(engine.num_bindings):
        is_input = engine.binding_is_input(idx)
        name = engine.get_binding_name(idx)
        op_type = engine.get_binding_dtype(idx)
        shape = engine.get_binding_shape(idx)
        if is_input:
            input_name.append(name)
        else:
            output_name.append(name)
        print('input id:',idx,'   is input: ', is_input,'  binding name:', name, '  shape:', shape, 'type: ', op_type)
    trt_model = TRTModule(engine, input_name, output_name)
    return trt_model

class OmniposeInfer():
    """OmniposeInfer is a class for inference of Omnipose/Cellpose model.
    """
    def __init__(self, config_yml: Dict) -> None: #define the model
        configure = Dict2ObjParser(config_yml).parse()
        model_name = configure.model.model_name
        cfg_path = configure.model.omnipose.config_path
        self.base_path = os.path.split(cfg_path)[0]
        with open(cfg_path, "r") as stream:
            cfg_yml = yaml.safe_load(stream)
            self.cfg = Dict2ObjParser(cfg_yml).parse()
        trt_path = configure.model.omnipose.model_path
        trt_model = create_trt_model(trt_path)
        parser = OmniposeParser(configure)
        model = parser.parse_model()
        model.net = trt_model
        self.model = model
        self.data_dir = self.cfg.data_path
        self.input_size = configure.data.input_size
        self.device = torch.device("cuda" if self.cfg.use_gpu else "cpu")

    def prepare_data(self):
        """
        prepare data for inference. 
        """
        #TODO: now the data structure is hard coded, need to be changed.
        self.files = io.get_image_files(os.path.join(self.data_dir,'im'), '_masks')
        mask_files = io.get_image_files(os.path.join(self.data_dir,'gt'), '_masks')
        self.images = [io.imread(f) for f in self.files] 
        # just for in house data:
        self.images = [img[:,:,0] for img in self.images]
        self.test_masks = [io.imread(f).astype(np.uint16) for f in mask_files]
               
    def save_result(self):
        """
        save the result of inference. Print the ap as well.
        """
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
   
    
    def calculate_infer_time(self, num: int):
        """calculating inference time using only patches, not the whole image. circulate num times, take the average.

        Args:
            num (int): number of patches to be inferenced.
        """
        infer_time = AverageMeter()
        infer_data = [torch.randn(1,*self.input_size,device = self.device) for _ in range(num)]
        for x in infer_data:
            end = time.time()
            y_hat = self.model.net(x)
            infer_time.update(time.time()-end)
        avg_infer_time = infer_time.avg
        print(f"average inference time is {avg_infer_time:.3f}")
    
    def calculate_energy(self, num: int): # the cpu/gpu energy consumed by the class.
        """calculate energy consumption using only patches, not the whole image. circulate num times, take the average. The value is based on codecarbon package.

        Args:
            num (int): number of patches to be inferenced.

        Returns:
            float: carbon dioxide emission in grams
        """
        infer_data = [torch.randn(1,*self.input_size,device = self.device) for _ in range(num)]
        self.model.net.to(self.device)
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
    """
    inference class for mmv_im2im model
    """
    def __init__(self, config_yml: Dict) -> None: #define the model
        configure = Dict2ObjParser(config_yml).parse()
        model_name = configure.model.model_name
        cfg_path = configure.model.mmv_im2im.config_path
        self.base_path = os.path.split(cfg_path)[0]
        trt_path = os.path.join(self.base_path,model_name+'.trt')
        trt_model = create_trt_model(trt_path)
        self.parser = Mmv_im2imParser(configure)
        model = self.parser.parse_model()
        model.net = trt_model
        self.model = model.net #
        self.config = self.parser.config
        self.data_cfg = self.config.data
        self.model_cfg = self.config.model
        self.device = torch.device('cuda')
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
    
    def save_result(self, pred: np.ndarray, out_fn) -> None:
        """save the result of one predicted image.

        Args:
            pred (np.ndarray): predicted np image
            out_fn (_type_): specify the output file name and path

        """
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
    
     
    def evaluate(self,
                 pred_dir: str,
                 gt_dir: str,
                 pred_data_type: str,
                 gt_data_type: str,
                 metric: Sequence[str]) -> None:
        """evaluation for mmv_im2im related tasks. Need to specify data location and data type.

        Args:
            pred_dir (str): location of the prediction
            gt_dir (str): location of the ground truth
            pred_data_type (str): prediction data type
            gt_data_type (str): ground truth data type
            metric (Sequence[str]): evaluation metrics.Currently support:
                1. SSIM for labelfree transformation
                2. Dice for semantic segmentation
                3. Pearson correlation for labelfree transformation
        """
        
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
          
    
    def calculate_infer_time(self,num: int) -> None: 
        """calculating inference time using only patches, not the whole image. circulate num times, take the average.

        Args:
            num (int): number of patches to be inferenced.
        """
        infer_time = AverageMeter()
        infer_data = [torch.randn(1,*self.input_size,device = self.device) for _ in range(num)]
        for x in infer_data:
            end = time.time()
            y_hat = self.model(x)
            infer_time.update(time.time()-end)
        avg_infer_time = infer_time.avg
        print(f"average inference time is {avg_infer_time:.3f}")

    def calculate_energy(self,num: int) -> float:
        """calculate energy consumption using only patches, not the whole image. circulate num times, take the average. The value is based on codecarbon package.

        Args:
            num (int): number of patches to be inferenced.

        Returns:
            float: carbon dioxide emission in grams
        """
        infer_data = [torch.randn(1,*self.input_size,device = self.device) for _ in range(num)]
        self.model.to(self.device) 
        tracker = EmissionsTracker(measure_power_secs = 1,
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
                x = x.unsqueeze(0).as_tensor().to(self.device)
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
                        device = torch.device('cpu'),
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
