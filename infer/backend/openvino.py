import numpy as np
import torch
from openvino.runtime import Core, get_version, PartialShape

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
    model = core.compile_model(opv_path, "CPU")
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
        outs = self.exec_net.infer({self.input_names[0]: inp})
        outs = {out.get_any_name(): torch.from_numpy(value) for out, value in outs.items()}
        return list(outs.values()) if len(outs.values())>1 else list(outs.values())[0]