import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
import mmv_im2im
from mmv_im2im.data_modules import get_data_module
from importlib import import_module
from mmv_im2im.configs.config_base import (
    ProgramConfig,
    configuration_validation,
)
from typing import Dict, List, Sequence, Text, Type, Union, TypeVar, Generic, Optional
from pyrallis import utils, cfgparsing
from pyrallis.parsers import decoding 
from mmv_im2im.utils.misc import generate_test_dataset_dict, parse_config
from mmv_im2im.utils.for_transform import parse_monai_ops_vanilla
import torch
from tqdm.contrib import tenumerate
from monai.transforms import RandSpatialCropSamples, Compose
from monai.data import DataLoader, Dataset
from .base import BaseParser

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
# currently no ground truth, so cannot be used for inference, just for calibration.
class Mmv_im2imDataset(Dataset):
    def __init__(self, data_cfg, transform=None):
        self.data_cfg = data_cfg
        self.dataset_list = generate_test_dataset_dict(self.data_cfg.inference_input.dir, self.data_cfg.inference_input.data_type)
        if self.data_cfg.preprocess is not None:
            self.pre_process = parse_monai_ops_vanilla(self.data_cfg.preprocess)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        img = AICSImage(self.dataset_list[idx]).reader.get_image_dask_data(**self.data_cfg.inference_input.reader_params)
        if self.pre_process is not None:
            img = self.pre_process(img)
        img = img.compute()
        img = torch.tensor(img.astype(np.float32))
        img = self.transform(img)
        return img

class Mmv_im2imParser(BaseParser):
    """parse the mmv_im2im model

    Args:
        Parser (_type_): base parser for the inherited parsers.
    """
    def __init__(self,config):
        super().__init__(config)
        self.cfg = parse_adaptor(config_class=ProgramConfig,config = self.meta_config.model.mmv_im2im.config_path)
        self.cfg = configuration_validation(self.cfg)
        self.data_cfg = self.cfg.data
        self.model_cfg = self.cfg.model
        # define variables
        self.model = None
        self.data = None
        self.pre_process = None

    @property
    def config(self):
        return self.cfg
            
    def parse_model(self):
        model_category = self.model_cfg.framework
        model_module = import_module(f"mmv_im2im.models.pl_{model_category}")
        my_model_func = getattr(model_module, "Model")
        self.model = my_model_func(self.model_cfg, train=False)
        pre_train = torch.load(self.model_cfg.checkpoint)
        self.model.load_state_dict(pre_train["state_dict"])
        self.model.eval()
        return self.model
    
    def parse_data(self):
        crop = RandSpatialCropSamples(roi_size = self.meta_config.data.input_size[-3:],
                                                   num_samples= 6,
                                                   random_size=False)
        dataset = Mmv_im2imDataset(self.data_cfg, transform = crop)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        return dataloader
        
        # create a dataloader and return:
        
 
    def calibrate(self,model,calib_num: int):
        """calibration step for the quantization to restore the precision. for each image, we crop 6 patches and feed them into the model to get the output.

        Args:
            model (_type_): model to be calibrated. Should be graph mode.
            calib_num (int): number of images to be used for calibration

        Raises:
            ValueError: _description_

        Returns:
            model: returned calibrated model
        """
        try:
            dataset = self.dataset_list[0:calib_num]
        except:
            raise ValueError(f'The number of calibrated images should between 0 to {self.dataset_length}')
        with torch.no_grad():
            for i, ds in tenumerate(dataset):
                img = AICSImage(ds).reader.get_image_dask_data(
                    **self.data_cfg.inference_input.reader_params
                )
                x = img.compute()
                x = torch.tensor(x.astype(np.float32))
                crop = RandSpatialCropSamples(roi_size = self.meta_config.data.input_size[-3:],
                                                   num_samples= 6,
                                                   random_size=False)
                if self.pre_process is not None:
                    x = self.pre_process(x)
                crop_list = crop(x)
                crop_list = [x.unsqueeze(0) for x in crop_list]
                for k in crop_list:
                    y_hat = model.net(k.as_tensor())
        print(f'--------------calibration done!----------')
        return model
