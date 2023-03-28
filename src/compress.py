import os
import shutil
from utils import Dict2ObjParser
import yaml
import numpy as np
import argparse
import torch
from src.parse_info import Mmv_im2imParser, OmniposeParser
from src.compress_ppl import Pipeline

_PARSER_DICT = dict(
        mmv_im2im =  lambda : Mmv_im2imParser,
        omnipose =  lambda: OmniposeParser
        )
    
def main():
    parser = argparse.ArgumentParser(description='Run the quantization')
    parser.add_argument('--cfg_path', type=str, default='configs/mmv_im2im/denoising_3d.yaml', help='config path.')
    parser.add_argument('--exp_path', type=str, default="experiment/denoising_trt_fp32", help='experiment path.')
    args = parser.parse_args()
    
    # ----------------------------------------------------------
    #1. Read the config file and set the data/model:
    # ----------------------------------------------------------
    with open(args.cfg_path, "r") as stream:
        config_yml = yaml.safe_load(stream)
        config = Dict2ObjParser(config_yml).parse()
    
    model_name = config.model.model_name
    exp_path = os.path.join(os.getcwd(),args.exp_path)
    os.makedirs(exp_path, exist_ok=True)
    config_path = config_yml['model'][model_name]['config_path']
    shutil.copy(config_path,exp_path)
    
    parser = _PARSER_DICT[model_name]()(config)
    model = parser.parse_model()
    data = parser.parse_data()
    
    # ----------------------------------------------------------
    #2. define and execute the pipeline:
    # ----------------------------------------------------------    
    pipeline = Pipeline.setup(config)
    pipeline(model, data, exp_path)
    
    # ----------------------------------------------------------
    #3. transform the model to IR compatible format:
    # openvino: .bin .xml
    # tensorrt: .engine
    # ----------------------------------------------------------       
    pipeline.network2ir()
    
if __name__ == "__main__":
    main()
