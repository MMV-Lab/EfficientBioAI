import os
import sys
import time
import glob
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
sys.path.append("..") 

import torch
import numpy as np
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from skimage.io import imsave as save_rgb
from codecarbon import EmissionsTracker
from torchmetrics import Dice, StructuralSimilarityIndexMeasure, PearsonCorrCoef
from monai.inferers import sliding_window_inference
from tqdm.contrib import tenumerate
from mmv_im2im.configs.config_base import (
    ProgramConfig,
    parse_adaptor,
    configuration_validation,
)
from mmv_im2im.data_modules import get_data_module
from mmv_im2im.utils.misc import generate_test_dataset_dict, parse_config
from mmv_im2im.utils.for_transform import parse_monai_ops_vanilla

from utils import Dict2ObjParser,AverageMeter,timer
from parse_info import Mmv_im2imParser
from .backend import create_opv_model, create_trt_model
from .base import BaseInfer

create_model = dict(openvino = create_opv_model,
                  tensorrt = create_trt_model)
device = dict(openvino = torch.device('cpu'),
              tensorrt = torch.device('cuda'))

def check_device(backend):
    if not torch.cuda.is_available() and backend == 'tensorrt':
        raise ValueError('TensorRT backend requires CUDA to be available')
    else:
        print('Using {} backend, device checked!'.format(backend))

class Mmv_im2imInfer(BaseInfer):
    """Inference class for Mmv_im2im model
    """
    def __init__(self, config_yml) -> None: #define the model
        super().__init__()
        configure = Dict2ObjParser(config_yml).parse()
        backend = configure.quantization.backend
        check_device(backend)
        cfg_path = configure.model.mmv_im2im.config_path
        self.base_path = os.path.split(cfg_path)[0]
        self.parser = Mmv_im2imParser(configure)
        infer_path = configure.model.mmv_im2im.model_path
        model = create_model[backend](infer_path)
        self.model = model
        self.config = self.parser.config
        self.data_cfg = self.config.data
        self.model_cfg = self.config.model
        self.device = device[backend]
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
    
    def save_result(self,pred: np.ndarray,out_fn) -> None:
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
                our_metric[k] = metric_table[k].to(torch.device('cpu')) #probably out of memory if use gpu
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