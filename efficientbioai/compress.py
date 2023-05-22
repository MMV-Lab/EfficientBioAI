import os
import shutil
from functools import partial
import yaml
import argparse
import torch
from parse_info import Mmv_im2imParser, OmniposeParser
from efficientbioai.utils.misc import Dict2ObjParser
from efficientbioai.compress_ppl import Pipeline
from efficientbioai.utils.logger import logger
import warnings

_PARSER_DICT = dict(
    mmv_im2im=lambda: Mmv_im2imParser, omnipose=lambda: OmniposeParser
)  # noqa: E501


def main():
    warnings.simplefilter(action="ignore")

    parser = argparse.ArgumentParser(description="Run the compression")
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="configs/omnipose/omnipose.yaml",
        help="config path.",
    )  # noqa: E501
    parser.add_argument(
        "--exp_path",
        type=str,
        default="experiment/test_cp_opv_fp32",
        help="experiment path.",
    )
    args = parser.parse_args()

    logger.info("=" * 40)  # split the log between different runs
    logger.info(f"start the experiment: {args.exp_path}")
    # ----------------------------------------------------------
    # 1. Read the config file and set the data/model:
    # ----------------------------------------------------------
    with open(args.cfg_path, "r") as stream:
        config_yml = yaml.safe_load(stream)
        config = Dict2ObjParser(config_yml).parse()

    model_name = config.model.model_name
    exp_path = os.path.join(os.getcwd(), args.exp_path)
    os.makedirs(exp_path, exist_ok=True)
    config_path = config_yml["model"][model_name]["config_path"]
    shutil.copy(config_path, exp_path)

    parser = _PARSER_DICT[model_name]()(config)
    model = parser.parse_model(
        device=torch.device("cpu")
    )  # prune and quantize can only be done on cpu
    data = parser.parse_data()
    calibrate = partial(parser.calibrate, args=parser.args)
    fine_tune = partial(parser.fine_tune, args=parser.args)

    # ----------------------------------------------------------
    # 2. define and execute the pipeline:
    # ----------------------------------------------------------
    pipeline = Pipeline.setup(config_yml)
    pipeline(model, data, fine_tune, calibrate, exp_path)

    # ----------------------------------------------------------
    # 3. transform the model to IR compatible format:
    # openvino: .bin .xml
    # tensorrt: .engine
    # ----------------------------------------------------------
    pipeline.network2ir()


if __name__ == "__main__":
    main()
