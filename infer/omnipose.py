import os
import time
import glob
import yaml
import sys
import logging as log
sys.path.append("..") 

import torch
import numpy as np
from cellpose import core, utils, io, models, metrics
from codecarbon import EmissionsTracker
from torchmetrics import Dice, StructuralSimilarityIndexMeasure, PearsonCorrCoef

from utils import Dict2ObjParser,AverageMeter,timer
from parse_info import OmniposeParser
from .backend import create_opv_model, create_trt_model

create_model = dict(openvino = create_opv_model,
                  tensorrt = create_trt_model)
device = dict(openvino = torch.device('cpu'),
              tensorrt = torch.device('cuda'))

def check_device(backend):
    if not torch.cuda.is_available() and backend == 'tensorrt':
        raise ValueError('TensorRT backend requires CUDA to be available')
    else:
        print('Using {} backend, device checked!'.format(backend))

class OmniposeInfer():
    
    def __init__(self, config_yml) -> None: #define the model
        configure = Dict2ObjParser(config_yml).parse()
        model_name = configure.model.model_name
        backend = configure.quantization.backend
        check_device(backend)
        cfg_path = config_yml['model'][model_name]['config_path']
        self.base_path = os.path.split(cfg_path)[0]
        infer_path  = configure.model.omnipose.model_path
        self.parser = OmniposeParser(configure)
        model = self.parser.parse_model()
        model.mkldnn = False
        model.net = create_model[backend](infer_path)
        self.model = model
        self.config = self.parser.config
        self.data_dir = self.config.data_path
        self.input_size = configure.data.input_size
        self.device = device[backend]
        
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
              tif=True
              )
        threshold = [0.5, 0.75, 0.9]
        ap,tp,fp,fn = metrics.average_precision(self.test_masks, self.masks, threshold=threshold)    
        print(ap)
    
    def calculate_infer_time(self,num: int) -> None: 
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
    
    def calculate_energy(self,num: int) -> float:
        """calculate energy consumption using only patches, not the whole image. circulate num times, take the average. The value is based on codecarbon package.

        Args:
            num (int): number of patches to be inferenced.

        Returns:
            float: carbon dioxide emission in grams
        """
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
                                  channels=self.config.channels,
                                  diameter=self.config.diameter,
                                  flow_threshold=self.config.flow_threshold,
                                  cellprob_threshold=self.config.cellprob_threshold,
                                  )
        self.masks = masks
        self.flows = flows
    
    def run_infer(self):
        self.prepare_data()
        self.core_infer()
        self.save_result()