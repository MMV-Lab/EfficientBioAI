import os
import sys
import yaml
import shutil
from typing import Optional, Sequence, Any, Union, Type

import torch
from tqdm.contrib import tenumerate
import numpy as np

from utils import Dict2ObjParser
from parse_info import Mmv_im2imParser, OmniposeParser
from torch.utils.data import DataLoader
from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup

class Pruner():
    """class for pruning algorithm.
    """
    def __init__(self,
                 model,
                 model_type,
                 pconfig
                 ):
        self.model = model
        self.model_type = model_type
        self.pconfig = pconfig
        
    def _get_network(self): 
        """extract the part in the torch module for quantization.
        """
        if self.model_type == 'mmv_im2im' or 'cellpose':
            self.network = self.model.net 
        elif self.model_type == 'academic':
            self.network = self.model
        else:
            raise NotImplementedError('model type not supported!')
        
    def _set_network(self):
        """retrive the quantized network and insert back to the original model.
        """
        if self.model_type == 'mmv_im2im' or 'cellpose':
            self.model.net = self.network
        elif self.model_type == 'academic':
            self.model = self.network
        else:
            raise NotImplementedError('model type not supported!')
        
    def _fine_tune(self):
        """fine-tune the quantized model to restore the accuracy. Only used in qat strategy.
        """
        #TODO: decouple.
        pass 
    
    def __call__(self,
                 input_size,
                 data: Union[DataLoader, Sequence[Any], None] = None,
                 ):
        if len(input_size) > 3:
            raise ValueError('currently cannot support 3d pruning!')
        self._get_network()
        dummy_input = torch.rand(1,*input_size).to(self.device)
        pruner = L1NormPruner(self.network, self.pconfig.config_list,mode='dependency_aware',dummy_input=dummy_input)
        pruner._unwrap_model()
        # compress the model and generate the masks
        _, masks = pruner.compress()
        # show the masks sparsity
        for name, mask in masks.items():
            print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()), ' shape : ', mask['weight'].shape)
        ModelSpeedup(self.network, torch.rand(1,*input_size).to(self.device), masks,customized_replace_func=self.pconfig.customized_replace_func).speedup_model()
        self._fine_tune()
        self._set_network()
        return self.model
        
    @staticmethod    
    def export_network(model,
                       model_type,                 
                       input_size,
                       input_names,
                       output_names,
                       output_path):
        io_names = [*input_names,*output_names]
        dynamic_axes = {k:{0: 'batch_size'} for k in io_names}
        torch.onnx.export(model,
                              torch.randn(1,*input_size),
                              os.path.join(output_path,f"{model_type}_deploy_model.onnx"),
                              opset_version = 11,
                              input_names = input_names,
                              output_names = output_names,
                              dynamic_axes = dynamic_axes
        )