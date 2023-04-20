import os
import yaml
import torch
from torch.utils.data import Dataset
from cellpose import io, models

from .base import BaseParser
from efficientbioai.utils import Dict2ObjParser


class OmniposeDataset(Dataset):
    def __init__(self, image_dir, mask_filter, transform=None):
        self.files = io.get_image_files(image_dir, mask_filter=mask_filter)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = io.imread(self.files[idx])
        if self.transform:
            image = self.transform(image)
        image = image[:, :, 0]  # only extract red channel
        return image


class OmniposeParser(BaseParser):
    """parse the omnipose model and data.

    Args:
        Parser (_type_): base parser for the inherited parsers.
    """

    def __init__(self, config):
        super().__init__(config)
        with open(self.meta_config.model.omnipose.config_path, "r") as stream:
            yml_file = yaml.safe_load(stream)
            self.args = Dict2ObjParser(yml_file).parse()

    @property
    def config(self):
        return self.args

    def parse_model(self):
        """parse cellpose/omnipose model. read the pretrained model if it exists.

        Returns:
            model: _description_
        """
        use_gpu = (
            self.meta_config.quantization.backend == "tensorrt"
            and torch.cuda.is_available()
        )
        self.device = torch.device("cuda" if use_gpu else "cpu")
        if self.args.pretrained_model is not None and os.path.exists(
            self.args.pretrained_model
        ):
            self.model = models.CellposeModel(
                gpu=use_gpu,
                pretrained_model=self.args.pretrained_model,
                device=self.device,
            )
            self.model.net.load_model(
                self.args.pretrained_model,
                #   cpu=not self.args.use_gpu,
                device=self.device,
            )
        else:
            self.model = models.CellposeModel(
                gpu=use_gpu,
                model_type=self.args.model_type,
                # omni=self.args.omni,
                dim=self.args.dim,
            )
        self.model.mkldnn = False  # use openvino/tensorrt backend instead of mkldnn
        self.model.net.mkldnn = False
        return self.model

    def parse_data(self):
        data = io.load_images_labels(self.args.compress_data_path)
        return data

    @staticmethod
    def fine_tune(model, data, device, args):
        model.train(*data, channels=args.channels)
        return model

    @staticmethod
    def calibrate(model, data, calib_num, device, args):
        model.net.to(device)
        with torch.no_grad():
            model.eval(
                data[0][0:calib_num],
                channels=args.channels,
                diameter=args.diameter,
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
            )
        return model.net
