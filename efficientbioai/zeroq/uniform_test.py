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
from utils import *
from omnipose_distill_data import *
from omnipose import Omnipose
import yaml
from collections import namedtuple 


class Dict2ObjParser:
    """Parse a nested dictionary into a nested named tuple."""

    def __init__(self, nested_dict):
        self.nested_dict = nested_dict

    def parse(self):
        nested_dict = self.nested_dict
        if (obj_type := type(nested_dict)) is not dict:
            raise TypeError(f"Expected 'dict' but found '{obj_type}'")
        return self._transform_to_named_tuples("root", nested_dict)

    def _transform_to_named_tuples(self, tuple_name, possibly_nested_obj):
        if type(possibly_nested_obj) is dict:
            named_tuple_def = namedtuple(tuple_name, possibly_nested_obj.keys())
            transformed_value = named_tuple_def(
                *[
                    self._transform_to_named_tuples(key, value)
                    for key, value in possibly_nested_obj.items()
                ]
            )
        elif type(possibly_nested_obj) is list:
            transformed_value = [
                self._transform_to_named_tuples(
                    f"{tuple_name}_{i}", possibly_nested_obj[i]
                )
                for i in range(len(possibly_nested_obj))
            ]
        else:
            transformed_value = possibly_nested_obj

        return transformed_value


# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--config_yml',
                        type=str,
                        default='/Users/zhouyu/Project/EfficientBioAI/configs/omnipose/instanceseg_2d.yaml',
                        help='path to config.yaml')
    parser.add_argument('--size',
                        type=int,
                        default=(8,2,224,224),
                        help='size of distilled data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    with open(args.config_yml, "r") as config:
        cfg_yml = yaml.safe_load(config)
        configure = Dict2ObjParser(cfg_yml).parse()
    # Load pretrained model
    model = Omnipose(configure)
    print('****** Full precision model loaded ******')

    # Load validation data
    test_image, test_mask = model.get_data()
    # Generate distilled data
    distil_data = getDistilData(model,args.size)
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
    # model.get_model().net = quantized_model
    model.infer()
    model.evaluate()
    model.save_result()
