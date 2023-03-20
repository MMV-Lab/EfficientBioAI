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

from utils import AverageMeter,timer
from .base import BaseInfer

class OmniposeInfer(BaseInfer):
    
    def __init__(self, config_yml) -> None: #define the model
        super().__init__(config_yml)
        model = self.parser.parse_model()
        model.mkldnn = False
        model.net = self.network
        self.model = model
        self.data_dir = self.config.data_path
        
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