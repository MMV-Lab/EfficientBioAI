from .openvino import OpenVINOModel
from .tensorrt import TRTModule

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

