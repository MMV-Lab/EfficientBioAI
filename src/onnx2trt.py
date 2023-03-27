import onnx
import pycuda.autoinit # noqa F401
import tensorrt as trt
import torch
import json
import pycuda.driver as cuda
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import argparse
import yaml
from src.utils import Dict2ObjParser

def onnx2trt(onnx_model,
             trt_path,
             log_level=trt.Logger.ERROR,
             max_workspace_size=1 << 30,
             device_id=0,
             mode='fp32',
             is_explicit=False,
             dynamic_range_file=None,
             quantization_config=None):
    
    if os.path.exists(trt_path):
        print(f'The "{trt_path}" exists. Remove it and continue.')
        os.remove(trt_path)

    device = torch.device('cuda:{}'.format(device_id))

    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'parse onnx failed:\n{error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    if mode == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        if dynamic_range_file:
            with open(dynamic_range_file, 'r') as f:
                dynamic_range = json.load(f)['tensorrt']['blob_range']

            for input_index in range(network.num_inputs):
                input_tensor = network.get_input(input_index)
                if input_tensor.name in dynamic_range:
                    amax = dynamic_range[input_tensor.name]
                    input_tensor.dynamic_range = (-amax, amax)
                    print(f'Set dynamic range of {input_tensor.name} as [{-amax}, {amax}]')

            for layer_index in range(network.num_layers):
                layer = network[layer_index]
                output_tensor = layer.get_output(0)
                if output_tensor.name in dynamic_range:
                    amax = dynamic_range[output_tensor.name]
                    output_tensor.dynamic_range = (-amax, amax)
                    print(f'Set dynamic range of {output_tensor.name} as [{-amax}, {amax}]')
        else:
            pass

    profile = builder.create_optimization_profile()
    profile.set_shape(quantization_config.data.io.input_names[0], 
                      tuple([quantization_config.data.dynamic_batch[0],*quantization_config.data.input_size]),
                      tuple([quantization_config.data.dynamic_batch[1],*quantization_config.data.input_size]),
                      tuple([quantization_config.data.dynamic_batch[2],*quantization_config.data.input_size])
                        ) #only suppport 1 input senario. for omnipose, the input io name is 'data', for mmv_im2im, the input io name is 'input'.
    print(quantization_config.data.input_size)
    print(quantization_config.data.dynamic_batch[0])
    config.add_optimization_profile(profile)
    engine = builder.build_engine(network, config)

    with open(trt_path, mode='wb') as f:
        f.write(bytearray(engine.serialize()))
    return engine


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Onnx to tensorrt')
    parser.add_argument('--onnx-path', type=str, default=None)
    parser.add_argument('--trt-path', type=str, default=None)
    parser.add_argument('--mode', choices=['fp32', 'int8'], default='int8')
    parser.add_argument('--clip-range-file', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    with open('configs/quantization_config.yaml', "r") as stream:
        quantization_config = yaml.safe_load(stream)
        quantization_config = Dict2ObjParser(quantization_config).parse()
        
    if args.onnx_path:
        onnx2trt(onnx_model = args.onnx_path,
                 trt_path=args.trt_path,
                 mode=args.mode,
                 log_level=trt.Logger.VERBOSE if args.verbose else trt.Logger.ERROR,
                 dynamic_range_file=args.clip_range_file,
                 quantization_config = quantization_config)