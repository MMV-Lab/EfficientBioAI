import os
import shutil
from functools import partial
import logging
import yaml
import argparse
from parse_info import Mmv_im2imParser, OmniposeParser
from efficientbioai.utils import Dict2ObjParser
from efficientbioai.compress_ppl import Pipeline
import warnings


_PARSER_DICT = dict(
    mmv_im2im=lambda: Mmv_im2imParser, omnipose=lambda: OmniposeParser
)  # noqa: E501


def main():
    warnings.simplefilter(action="ignore")
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )  # five levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    # will only log the warning and above

    parser = argparse.ArgumentParser(description="Run the quantization")
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
    model = parser.parse_model()
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
