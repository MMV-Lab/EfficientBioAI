import os
from typing import Optional, Sequence, Any, Union, Type
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import BackendType, prepare_by_platform
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.advanced_ptq import ptq_reconstruction
from tqdm.contrib import tenumerate

_BACKEND = dict(
                tensorrt = BackendType.Tensorrt,
                openvino = BackendType.OPENVINO,
                )

class Quantizer():
    """Quantizer class for quantizing a model.
    """
    def __init__(self,
                 model,
                 model_type,
                 qconfig= None,
                 device = None
                 ):
        self.model = model
        self.model_type = model_type
        self.qconfig = qconfig
        self.device = device
        
    def _get_network(self): 
        """extract the part in the torch module for quantization.
        """
        if self.model_type in ['mmv_im2im', 'cellpose', 'omnipose']:
            self.network = self.model.net 
        elif self.model_type == 'academic':
            self.network = self.model
        else:
            raise NotImplementedError('model type not supported!')
        
    def _set_network(self):
        """retrive the quantized network and insert back to the original model.
        """
        if self.model_type in ['mmv_im2im', 'cellpose', 'omnipose']:
            self.model.net = self.network
        elif self.model_type == 'academic':
            self.model = self.network
        else:
            raise NotImplementedError('model type not supported!')    
    
    def __call__(self,
                 input_size,
                 input_names,
                 output_names,
                 output_path,
                 data: Union[DataLoader, Sequence[Any], None] = None,
                 calibrate = None
                 ):
        input_shape = {input_names[0]: [1,*input_size]} # batchsize+channel, ZYX. only consider 1 input senario.
        io_names = [*input_names,*output_names]
        dynamic_axes = {k:{0: 'batch_size'} for k in io_names}
        backend = _BACKEND[self.qconfig['backend']]
        self._get_network()
        if self.qconfig['run_mode'] == 'int8':
            #since additional model type can only accept tuple. so we need to convert it to tuple. another way is to directly use tuple in config file.[https://itecnote.com/tecnote/python-how-to-add-a-python-tuple-to-a-yaml-file-using-pyyaml/]
            extra_config = {
                            "extra_qconfig_dict": self.qconfig['extra_config']['extra_qconfig_dict'],
                            "extra_quantizer_dict": { "additional_module_type": (torch.nn.Conv3d, torch.nn.MaxPool3d,torch.nn.ConvTranspose3d), "additional_function_type": [torch.cat,]} 
                            }
            self.network = prepare_by_platform(self.network, backend, extra_config) # trace model and add quant nodes for model on backend
            enable_calibration(self.network) # turn on calibration, ready for gathering data
            self._set_network()
            self.network = calibrate(self.model, data, 4, self.device) # run calibration
            self._get_network()
            enable_quantization(self.network) # turn on actually quantization, ready for simulating Backend inference
            extra_kwargs = dict(input_names = input_names,
                                output_names = output_names,
                                dynamic_axes = dynamic_axes
                                )
            convert_deploy(self.network,
                           backend,
                           input_shape,
                           model_name = self.model_type,
                           output_path = output_path,
                           deploy_to_qlinear = False,
                           **extra_kwargs
                           )
        else: #run in fp32 mode.
            torch.onnx.export(self.network,
                              torch.randn(1,*input_size),
                              os.path.join(output_path,f"{self.model_type}_deploy_model.onnx"),
                              opset_version = 11,
                              input_names = input_names,
                              output_names = output_names,
                              dynamic_axes = dynamic_axes
            )
        
        
        
        
            
    