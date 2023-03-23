import os
import sys
import yaml
import shutil

from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import BackendType, prepare_by_platform
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.advanced_ptq import ptq_reconstruction
import torch
from tqdm.contrib import tenumerate
import numpy as np

from utils import Dict2ObjParser
from parse_info import Mmv_im2imParser, OmniposeParser
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run the quantization')
    parser.add_argument('--cfg_path', type=str, default='configs/mmv_im2im/denoising_3d.yaml', help='config path.')
    parser.add_argument('--exp_path', type=str, default="experiment/denoising_trt_fp32", help='experiment path.')
    args = parser.parse_args()
    # ----------------------------------------------------------
    #1. Read the config file and set the data/model:
    # ----------------------------------------------------------
    with open(args.cfg_path, "r") as stream:
        quantization_config_yml = yaml.safe_load(stream)
        quantization_config = Dict2ObjParser(quantization_config_yml).parse()
    
    model_cfg = quantization_config.model
    quan_cfg = quantization_config.quantization
    data_cfg = quantization_config.data
    exp_path = os.path.join(os.getcwd(),args.exp_path)
    os.makedirs(exp_path, exist_ok=True)
    config_path = quantization_config_yml['model'][model_cfg.model_name]['config_path']
    shutil.copy(config_path,exp_path)
    
    parser_dict = dict(
        mmv_im2im =  lambda : Mmv_im2imParser,
        omnipose =  lambda: OmniposeParser
    )
    parser = parser_dict[model_cfg.model_name]()(quantization_config)
    model = parser.parse_model()
    parser.parse_data()
    
    # ----------------------------------------------------------
    #2. Quantize the model:
    # ----------------------------------------------------------
    input_shape = {data_cfg.io.input_names[0]:[1,*data_cfg.input_size]} # batchsize+channel, ZYX. only consider 1 input senario. only for omnipose
    io_names = [*data_cfg.io.input_names,*data_cfg.io.output_names]
    dynamic_axes = {k:{0: 'batch_size'} for k in io_names}
    supported_backend = dict(
                            tensorrt = BackendType.Tensorrt,
                            openvino = BackendType.OPENVINO,
                            )
    # learn more about qconfigs from https://github.com/ModelTC/MQBench/blob/bbf54367180a9e5e5ce37efcd7b8d7ebb5b926eb/docs/source/user_guide/internal/learn_config.rst
    extra_config = {
    "extra_qconfig_dict": quantization_config_yml['quantization']['extra_qconfig_dict'],
    "extra_quantizer_dict": { "additional_module_type": (torch.nn.Conv3d, torch.nn.MaxPool3d,torch.nn.ConvTranspose3d), "additional_function_type": [torch.cat,]} 
    }
    backend = supported_backend[quan_cfg.backend]

    if quantization_config.quantization.run_mode == 'int8':
        model.net = prepare_by_platform(model.net, backend,extra_config) # trace model and add quant nodes for model on backend
        enable_calibration(model.net) # turn on calibration, ready for gathering data
        model = parser.calibrate(model,calib_num=4)
        enable_quantization(model.net) # turn on actually quantization, ready for simulating Backend inference
        extra_kwargs = dict(input_names=data_cfg.io.input_names, output_names=data_cfg.io.output_names, dynamic_axes = dynamic_axes)
        convert_deploy(model.net, backend, input_shape,model_name = model_cfg.model_name,output_path = exp_path,deploy_to_qlinear = False,**extra_kwargs)
    else:
        # device = torch.device("cpu" if quantization_config.quantization.backend == 'openvino' else "cuda")
        torch.onnx.export(model.net, torch.randn(1,*data_cfg.input_size), os.path.join(exp_path,f"{model_cfg.model_name}_deploy_model.onnx"), opset_version=11, input_names=data_cfg.io.input_names, output_names=data_cfg.io.output_names, dynamic_axes = dynamic_axes)
    # ----------------------------------------------------------
    # 3. transform onnx to file compatible with infererence engine
    #    openvino: .bin .xml
    #    tensorrt: .engine
    # ----------------------------------------------------------
    if quan_cfg.backend == 'tensorrt':
        from onnx2trt import onnx2trt
        trt_path = os.path.join(exp_path,f"{model_cfg.model_name}.trt")
        dynamic_file_path = os.path.join(exp_path,f"{model_cfg.model_name}_clip_ranges.json")
        print(os.path.join(exp_path,f"{model_cfg.model_name}_deploy_model.onnx"))
        onnx2trt(onnx_model = os.path.join(exp_path,f"{model_cfg.model_name}_deploy_model.onnx"),
                trt_path = trt_path,
                mode = quan_cfg.run_mode,
                dynamic_range_file = dynamic_file_path,
                quantization_config = quantization_config)
        print('transform done!')
        # save the config file to the folder:
        quantization_config_yml['model'][model_cfg.model_name]['model_path'] = trt_path
        quantization_config_yml['quantization']['dynamic_range_file'] = dynamic_file_path
        quantization_config_yml['model'][model_cfg.model_name]['config_path'] = os.path.join(exp_path,os.path.basename(config_path))
        with open(os.path.join(exp_path,f"{model_cfg.model_name}.yaml"),'w') as stream:
            yaml.dump(quantization_config_yml, stream)

    else: #openvino
        import subprocess
        subprocess.run(["mo", "--input_model", f"{exp_path}/{model_cfg.model_name}_deploy_model.onnx"])
        xml_path = os.path.join(exp_path,f"{model_cfg.model_name}_deploy_model.xml")
        quantization_config_yml['model'][model_cfg.model_name]['model_path'] = xml_path
        quantization_config_yml['model'][model_cfg.model_name]['config_path'] = os.path.join(exp_path,os.path.basename(config_path))
        with open(os.path.join(exp_path,f"{model_cfg.model_name}.yaml"),'w') as stream:
            yaml.dump(quantization_config_yml, stream)
        for ext in ['bin','mapping','xml']:
            shutil.move(os.path.join(os.getcwd(),f'{model_cfg.model_name}_deploy_model.{ext}'),
                        os.path.join(exp_path,f'{model_cfg.model_name}_deploy_model.{ext}')
                        )
  
if __name__ == '__main__':
    main()
