import torch
from typing import Any, List, Union
from pathlib import Path
import tensorrt as trt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = trt.Logger(trt.Logger.INFO)


def trt_version():
    return trt.__version__


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= "7.0" and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


class TRTModule(torch.nn.Module):
    """TRTModule is a wrapper of TensorRT engine. Behave like a torch.nn.Module"""

    def __init__(
        self,
        engine: Any = None,
        input_names: List[str] = None,
        output_names: List[str] = None,
    ):
        super(TRTModule, self).__init__()
        self.engine = engine
        if self.engine is not None:
            # engine create and execute the context
            self.context = self.engine.create_execution_context()
            self.context.active_optimization_profile = 0

        self.input_names = input_names
        self.output_names = output_names

    def forward(self, *inputs):
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            # set the shape
            self.context.set_binding_shape(idx, tuple(inputs[i].shape))
            bindings[idx] = inputs[i].contiguous().data_ptr()

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)

        outputs = tuple(outputs)
        return outputs[0] if len(outputs) == 1 else reversed(outputs)


def create_trt_model(trt_path: Union[str, Path]) -> TRTModule:
    """Create a trt model from a trt file.

    Args:
        trt_path (Union[str, Path]): path to the trt engine file

    Returns:
        TRTModule: trt engine
    """
    with open(trt_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    input_name = []
    output_name = []
    for idx in range(engine.num_bindings):
        is_input = engine.binding_is_input(idx)
        name = engine.get_binding_name(idx)
        op_type = engine.get_binding_dtype(idx)
        shape = engine.get_binding_shape(idx)
        if is_input:
            input_name.append(name)
        else:
            output_name.append(name)
        print(
            "input id:",
            idx,
            "   is input: ",
            is_input,
            "  binding name:",
            name,
            "  shape:",
            shape,
            "type: ",
            op_type,
        )
    trt_model = TRTModule(engine, input_name, output_name)
    return trt_model
