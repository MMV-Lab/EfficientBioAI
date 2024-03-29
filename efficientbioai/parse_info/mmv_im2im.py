import numpy as np
from pathlib import Path
import shutil
from aicsimageio import AICSImage
from importlib import import_module
from mmv_im2im.configs.config_base import (
    ProgramConfig,
    configuration_validation,
)
from typing import Dict, List, Sequence, Text, Type, Union, TypeVar, Generic, Optional
from pyrallis import utils, cfgparsing
from pyrallis.parsers import decoding
from mmv_im2im.utils.misc import generate_test_dataset_dict, parse_ops_list
from mmv_im2im.utils.for_transform import parse_monai_ops_vanilla
from mmv_im2im.data_modules import get_data_module
import torch
from tqdm.contrib import tenumerate
from monai.transforms import RandSpatialCropSamples
from monai.data import DataLoader, Dataset
import pytorch_lightning as pl
from .base import BaseParser
from efficientbioai.utils.logger import logger

T = TypeVar("T")


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


# TODO: https://stackoverflow.com/questions/59467781/pytorch-dataloader-for-image-gt-dataset
# currently no ground truth, so cannot be reused for inference, just for calibration.
class Mmv_im2imDataset(Dataset):
    def __init__(self, data_cfg, transform=None):
        self.data_cfg = data_cfg
        self.dataset_list = generate_test_dataset_dict(
            self.data_cfg.inference_input.dir, self.data_cfg.inference_input.data_type
        )
        if self.data_cfg.preprocess is not None:
            self.pre_process = parse_monai_ops_vanilla(self.data_cfg.preprocess)
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        img = AICSImage(self.dataset_list[idx]).reader.get_image_dask_data(
            **self.data_cfg.inference_input.reader_params
        )
        img = img.compute()
        img = torch.tensor(img.astype(np.float32))
        if self.pre_process is not None:
            img = self.pre_process(img)
        img = self.transform(img)
        return img


class Mmv_im2imParser(BaseParser):
    """parse the mmv_im2im model

    Args:
        Parser (_type_): base parser for the inherited parsers.
    """

    def __init__(self, config):
        super().__init__(config)
        self.args = parse_adaptor(
            config_class=ProgramConfig,
            config=self.meta_config.model.mmv_im2im.config_path,
        )
        self.args = configuration_validation(self.args)
        self.data_cfg = self.args.data
        self.model_cfg = self.args.model
        self.train_cfg = self.args.trainer
        # define variables
        self.model = None
        self.data = None
        self.pre_process = None

    @property
    def config(self):
        return self.args

    def parse_model(self, device=torch.device("cuda")):
        model_category = self.model_cfg.framework
        model_module = import_module(f"mmv_im2im.models.pl_{model_category}")
        my_model_func = getattr(model_module, "Model")
        self.model = my_model_func(self.model_cfg, train=True)
        pre_train = torch.load(self.model_cfg.checkpoint)
        self.model.load_state_dict(pre_train["state_dict"])
        return self.model

    def parse_data(self):
        # crop = RandSpatialCropSamples(
        #     roi_size=self.meta_config.data.input_size[-3:],
        #     num_samples=6,
        #     random_size=False,
        # )
        # dataset = Mmv_im2imDataset(self.data_cfg, transform=crop)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        # return dataloader
        data = get_data_module(self.data_cfg)
        return data

    @staticmethod
    def fine_tune(model, data, device, args):
        # set up training
        device = "cpu" if device == torch.device("cpu") else "gpu"
        params = {
            "accelerator": device,
            "max_epochs": 10,
            "precision": 16,
            "devices": 1,
            "limit_val_batches": 0,
            "detect_anomaly": True,
        }  # validation sometimes may cause error
        trainer = pl.Trainer(callbacks=[], **params)
        logger.info("Start fine tuning...")
        trainer.fit(model=model, datamodule=data)
        logger.info("Fine tuning finished.")
        shutil.rmtree("tmp/")
        return model.net

    @staticmethod
    def calibrate(model, data, calib_num=4, device=torch.device("cpu"), args=None):
        model.eval()
        data.setup()
        dataloader = data.train_dataloader()
        with torch.no_grad():
            for i, x in tenumerate(dataloader):
                model.net(x["IM"].as_tensor())
                if i >= calib_num:
                    break
        return model.net
