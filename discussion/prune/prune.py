import os
import shutil
from functools import partial
import yaml
from pathlib import Path
from efficientbioai.parse_info import Mmv_im2imParser, OmniposeParser
from efficientbioai.infer import Mmv_im2imInfer, OmniposeInfer
from efficientbioai.utils.misc import Dict2ObjParser
from efficientbioai.compress_ppl import Pipeline
from efficientbioai.utils.logger import logger

parser_dict = dict(
    mmv_im2im=Mmv_im2imParser,
    omnipose=OmniposeParser,
)  # noqa: E501
infer_dict = dict(
    mmv_im2im=Mmv_im2imInfer,
    omnipose=OmniposeInfer,
)
sparsity_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pruner_type = ["L1NormPruner", "L2NormPruner", "FPGMPruner"]
cfg_path = Path("/home/ISAS.DE/yu.zhou/EfficientBioAI/discussion/prune/omnipose.yaml")
# ----------------------------------------------------------
# 1. Read the config file and set the data/model:
# ----------------------------------------------------------
with open(cfg_path, "r") as stream:
    config_yml = yaml.safe_load(stream)
    config = Dict2ObjParser(config_yml).parse()

for type in pruner_type:
    for ratio in sparsity_ratio:
        exp_path = (
            f"experiment/discussion/prune/{config.model.model_name}/{type}/{ratio}"
        )
        logger.info("=" * 40)  # split the log between different runs
        logger.info(
            f"start the experiment: {exp_path},with sparsity ratio: {ratio}, pruner type: {type}"
        )
        model_name = config.model.model_name
        exp_path = os.path.join(os.getcwd(), exp_path)
        os.makedirs(exp_path, exist_ok=True)
        config_path = config_yml["model"][model_name]["config_path"]
        shutil.copy(config_path, exp_path)

        parser = parser_dict[model_name](config)
        model = parser.parse_model()
        data = parser.parse_data()
        calibrate = partial(parser.calibrate, args=parser.args)
        fine_tune = partial(parser.fine_tune, args=parser.args)

        # ----------------------------------------------------------
        # 2. define and execute the pipeline:
        # ----------------------------------------------------------
        config_yml["prune"]["type"] = type
        config_yml["prune"]["config_list"][0]["sparsity_per_layer"] = ratio
        pipeline = Pipeline.setup(config_yml)
        pipeline(model, data, fine_tune, calibrate, exp_path)

        # ----------------------------------------------------------
        # 3. transform the model to IR compatible format:
        # openvino: .bin .xml
        # tensorrt: .engine
        # ----------------------------------------------------------
        pipeline.network2ir()
        # ----------------------------------------------------------
        # 4. Do the inference:
        # ----------------------------------------------------------
        config_yml["data"]["save_path"] = exp_path
        infer = infer_dict[model_name](config_yml=config_yml)
        infer.run_infer()
        infer.calculate_latency_energy()