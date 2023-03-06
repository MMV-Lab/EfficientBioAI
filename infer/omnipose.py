import os
import time
import glob
import numpy as np
import torch
import yaml
import sys
import logging as log
from cellpose import core, utils, io, models, metrics
from utils import Dict2ObjParser,AverageMeter
from parse_info import Mmv_im2imParser, OmniposeParser
from backend import TRTModule,OpenVINOModel

inf_engine = dict(openvino = OpenVINOModel,
                  tensorrt = TRTModule)

class OmniposeInfer():
    
    def __init__(self, config_yml) -> None: #define the model
        configure = Dict2ObjParser(config_yml).parse()
        model_name = configure.model.model_name
        backend = configure.quantization.backend
        cfg_path = config_yml['model'][model_name]['config_path']
        self.base_path = os.path.split(cfg_path)[0]
        with open(cfg_path, "r") as stream:
            cfg_yml = yaml.safe_load(stream)
            self.cfg = Dict2ObjParser(cfg_yml).parse()
        infer_path = self.cfg.model_path
        self.parser = OmniposeParser(configure)
        model = self.parser.parse_model()
        model.mkldnn = False
        model.net.mkldnn = False
        model.net = OpenVINOModel(model.net,config_yml)
        self.model = model
        self.data_dir = self.cfg.data_path
        
        
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
    
    def run_infer(self):
        self.prepare_data()
        masks, flows, _ = self.model.eval(self.images, 
                                  channels=self.cfg.channels,
                                  diameter=self.cfg.diameter,
                                  flow_threshold=self.cfg.flow_threshold,
                                  cellprob_threshold=self.cfg.cellprob_threshold,
                                  )
        self.masks = masks
        self.flows = flows
        self.save_result()