import os
import yaml
import shutil

import torch

from .quantizer import Quantizer
from .pruner import Pruner
from utils import Dict2ObjParser

class Pipeline():
    """class for generating pipeline for model compression.
    """
    def __init__(self,
                 config_dict,
                 prune = False,
                 quantize = True):
        self.config_dict = config_dict
        self.prune = prune
        self.quantize = quantize
        
    @classmethod
    def setup(cls, config_dict):
        if 'prune' in config_dict and config_dict['quantization']['run_mode'] == 'int8':
            return cls(config_dict, True, True)
        elif 'prune' in config_dict and config_dict['quantization']['run_mode'] == 'fp32':
            return cls(config_dict, True, False)
        elif 'prune' not in config_dict and config_dict['quantization']['run_mode'] == 'int8':
            return cls(config_dict, False, True)
        elif 'prune' not in config_dict and config_dict['quantization']['run_mode'] == 'fp32':
            return cls(config_dict, False, False)
        else:
            raise NotImplementedError('compression strategy not supported!')
    
    def __call__(self,model,data, path):
        
        self.config = Dict2ObjParser(self.config_dict).parse() 
        self.input_size = self.config.data.input_size
        self.model_name = self.config.model.model_name
        self.config_path = self.config_dict['model'][self.model_name]['config_path']
        self.input_names = self.config.data.io.input_names
        self.output_names = self.config.data.io.output_names
        self.output_path = path
        self.backend = self.config.quantization.backend
        self.run_mode = self.config.quantization.run_mode
        self.dynamic_batch = self.config.data.dynamic_batch
        
        quantizer = Quantizer(model, self.model_name, self.config.quantization, device = None)
        if self.prune:
            pruner = Pruner(model,self.model_name, self.config.prune)
            model = pruner(self.input_size, data)
        quantizer = Quantizer(model, self.model_name, self.config.quantization, device = None)
        quantizer(self.input_size, self.input_names, self.output_names, self.output_path, data)
    

    def network2ir(self):
        if self.backend == 'tensorrt':
            try:
                from src.onnx2trt import onnx2trt
            except:
                raise ImportError('tensorrt/pycuda not correctly installed!')
            trt_path = os.path.join(self.output_path,f"{self.model_name}.trt")
            dynamic_file_path = os.path.join(self.output_path,f"{self.model_name}_clip_ranges.json")
            print(os.path.join(self.output_path,f"{self.model_name}_deploy_model.onnx"))
            onnx2trt(onnx_model = os.path.join(self.output_path,f"{self.model_name}_deploy_model.onnx"),
                    trt_path = trt_path,
                    mode = self.run_mode,
                    dynamic_range_file = dynamic_file_path,
                    input_names = self.input_names,
                    input_size = self.input_size,
                    dynamic_batch = self.dymamic_batch
                    )
            print('transform done!')
            # save the config file to the folder:
            self.config_dict['model'][self.model_name]['model_path'] = trt_path
            self.config_dict['quantization']['dynamic_range_file'] = dynamic_file_path
            self.config_dict['model'][self.model_name]['config_path'] = os.path.join(self.output_path,os.path.basename(self.config_path))
            with open(os.path.join(self.output_path,f"{self.model_name}.yaml"),'w') as stream:
                yaml.dump(self.config_dict, stream)

        elif self.backend == 'openvino': #openvino
            import subprocess
            try:
                subprocess.run(["mo", "--input_model", f"{self.output_path}/{self.model_name}_deploy_model.onnx"])
            except:
                raise ImportError('Please check openvino model optimizer installation path!')
            xml_path = os.path.join(self.output_path,f"{self.model_name}_deploy_model.xml")
            self.config_dict['model'][self.model_name]['model_path'] = xml_path
            self.config_dict['model'][self.model_name]['config_path'] = os.path.join(self.output_path,os.path.basename(self.config_path))
            with open(os.path.join(self.output_path,f"{self.model_name}.yaml"),'w') as stream:
                yaml.dump(self.config_dict, stream)
            for ext in ['bin','mapping','xml']:
                shutil.move(os.path.join(os.getcwd(),f'{self.model_name}_deploy_model.{ext}'),
                            os.path.join(self.output_path,f'{self.model_name}_deploy_model.{ext}')
                            )      
            print('transform done!')
                
        else:
            raise NotImplementedError('backend not supported!')
        
        