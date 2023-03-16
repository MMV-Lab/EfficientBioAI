import numpy as np
import torch
from openvino.runtime import Core, get_version, PartialShape
from utils import Dict2ObjParser

def create_opv_model(opv_path):
    """create opv model

    Args:
        opv_path (_type_): _description_

    Returns:
        _type_: _description_
        
    Notes:
        model: <class 'openvino.runtime.ie_api.CompiledModel'>
        model.inputs: List(<class 'openvino.pyopenvino.ConstOutput'>) 
            - get_shape()
            - get_any_name()
    """
    core = Core()
    config = {"PERFORMANCE_HINT": "THROUGHPUT"}
    model = core.compile_model(opv_path,'CPU',config)
    input_names = [model.inputs[i].get_any_name() for i in range(len(model.inputs))]
    output_names = [model.outputs[i].get_any_name() for i in range(len(model.outputs))]
    opv_model = OPVModule(engine=model, input_names=input_names, output_names=output_names)
    return opv_model
    

class OPVModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(OPVModule, self).__init__()
        self.engine = engine
        self.exec_net = engine.create_infer_request()
        self.input_names = input_names
        self.output_names = output_names
    
    def forward(self,inp):
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