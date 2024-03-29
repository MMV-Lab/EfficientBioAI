import yaml
import argparse

from efficientbioai.utils.misc import Dict2ObjParser
from infer import Mmv_im2imInfer, OmniposeInfer

parser = argparse.ArgumentParser(description="Run the inference")
parser.add_argument(
    "--config",
    type=str,
    default="experiment/labelfree_opv_int8/mmv_im2im.yaml",
    help="path to the config file.",
)
args = parser.parse_args()

cfg_path = args.config

with open(cfg_path, "r") as stream:
    cfg_yml = yaml.safe_load(stream)
    cfg = Dict2ObjParser(cfg_yml).parse()
model_name = cfg.model.model_name
infer_dict = dict(mmv_im2im=Mmv_im2imInfer, omnipose=OmniposeInfer)

inference = infer_dict[model_name](config_yml=cfg_yml)

# for mmv_im2im:
inference.run_infer()
inference.evaluate(
    "/home/ISAS.DE/yu.zhou/EfficientBioAI/data/mmv_im2im/denoise/test/pred",
    "/home/ISAS.DE/yu.zhou/EfficientBioAI/data/mmv_im2im/denoise/test/gt",
    ".tif",
    ".tif",
    ["Dice"],
)  # currently only for mmv_im2im # noqa: E501
# inference.calculate_infer_time(num= 100)
# inference.calculate_energy(num=100)
# inference.calculate_energy_latency(num=100, image_size=[95, 1024, 1024])

# for omnipose:
# inference.run_infer()
# inference.calculate_infer_time(num=1000)
# inference.evaluate()
# inference.calculate_energy(num=1000)
