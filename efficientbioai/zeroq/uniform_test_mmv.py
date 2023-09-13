#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import argparse
import torch
from pathlib import Path
from utils import *
from mmv_distill_data import *
from mmv import Mmv
import yaml
from collections import namedtuple 
from typing import Dict, List, Sequence, Text, Type, Union, TypeVar, Generic, Optional
from pyrallis import utils, cfgparsing
from pyrallis.parsers import decoding
from mmv_im2im.configs.config_base import (
    ProgramConfig,
    configuration_validation,
)
T = TypeVar("T")

# mmv settings
def parse_adaptor(
    config_class: Type[T],
    config: Optional[Union[Path, str]] = None,
    args: Optional[Sequence[str]] = None,
) -> T:
    file_args = cfgparsing.load_config(open(config, "r"))
    file_args = utils.flatten(file_args, sep=".")
    parsed_arg_values = file_args
    deflat_d = utils.deflatten(parsed_arg_values, sep=".")
    cfg = decoding.decode(config_class, deflat_d)
    return cfg

# script settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--config_yml',
                        type=str,
                        default='/Users/zhouyu/Project/EfficientBioAI/configs/mmv_im2im/labelfree_3d_inference.yaml',
                        help='path to config.yaml')
    parser.add_argument('--size',
                        type=int,
                        default=(1,1,32,128,128),
                        help='size of distilled data, should be BC(Z)YX')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    mmv_args = parse_adaptor(
            config_class=ProgramConfig,
            config=args.config_yml,
        )
    mmv_args = configuration_validation(mmv_args)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    device = torch.device('cuda')
    model = Mmv(mmv_args)
    print('****** Full precision model loaded ******')

    # Generate distilled data
    distil_data = getDistilData(model,args.size, device= device)
    print('****** Data loaded ******')

    # Quantize single-precision model to 8-bit model
    quantized_model = quantize_model(model.get_model().net)
    # Freeze BatchNorm statistics
    quantized_model.eval()

    # Update activation range according to distilled data
    update(quantized_model, distil_data)
    print('****** Zero Shot Quantization Finished ******')

    # Freeze activation range during test
    freeze_model(quantized_model)

    # Test the final quantized model
    model.get_model().net = quantized_model
    model.infer()
    model.evaluate(metric=['SSIM'])
