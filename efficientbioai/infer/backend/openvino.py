import torch
from typing import Any, List, Union
from pathlib import Path
from openvino.runtime import Core


class OPVModule(torch.nn.Module):
    """openvino infer request model"""

    def __init__(
        self,
        engine: Any = None,
        input_names: List[str] = None,
        output_names: List[str] = None,
    ):
        super(OPVModule, self).__init__()
        self.engine = engine
        self.exec_net = engine.create_infer_request()
        self.input_names = input_names
        self.output_names = output_names

    def forward(self, inp):
        outs = self.exec_net.infer({self.input_names[0]: inp})
        outs = {
            out.get_any_name(): torch.from_numpy(value) for out, value in outs.items()
        }
        return list(outs.values()) if len(outs.values()) > 1 else list(outs.values())[0]


def create_opv_model(opv_path: Union[str, Path]) -> OPVModule:
    """create opv model
    Args:
        opv_path (Union[str, Path]): path to load the openvino ir model
    Returns:
        OPVModule: an opv infer request

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
    opv_model = OPVModule(
        engine=model, input_names=input_names, output_names=output_names
    )
    return opv_model
