import numpy as np
import torch
from openvino.runtime import Core, get_version, PartialShape
from utils import Dict2ObjParser

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
    def __init__(self,config_yml):
        configure = Dict2ObjParser(config_yml).parse()
        model_name = configure.model.model_name
        self._nets = {}
        self._model_id = "default"
        self.infer_path = config_yml['model'][model_name]['model_path']
        self.input_names = configure.data.io.input_names
        self.output_names = configure.data.io.output_names
        self.exec_net = self._init_model()

    def _init_model(self):
        if self._model_id in self._nets:
            return self._nets[self._model_id]
        self.opv_model = create_model(self.infer_path)
        infer_request = self.opv_model.create_infer_request()
        self._nets[self._model_id] = infer_request
        return infer_request


    def __call__(self, inp):
        # exec_net = self._init_model(inp)
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