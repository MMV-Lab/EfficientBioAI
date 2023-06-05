import matplotlib.pyplot as plt
import numpy as np
from unet.model import UNet
from deconoising import utils
from deconoising import training
from tifffile import imread

# See if we can use a GPU

import torch
from tifffile import imread, imsave
from scipy.ndimage import gaussian_filter

from functools import partial
from pathlib import Path
import yaml
from copy import deepcopy
from efficientbioai.compress_ppl import Pipeline
from efficientbioai.utils.misc import Dict2ObjParser
from deconoising import prediction

device = torch.device("cpu")

path = "./data/Convallaria_diaphragm/"
fileName = "20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif"
dataName = "convallaria"  # This will be used to name the network model


data_image = imread(path + fileName)
nameModel = dataName + "_network_example"  # specify the name of your network
meanValue = 520.0
data = np.array(data_image).astype(np.float32)
data = data - meanValue


def calibrate(
    model,
    dataloader,
    device=torch.device("cpu"),
):
    model.eval()
    model.to(device)
    for data in dataloader:
        prediction.tiledPredict(model, data, ps=256, overlap=24, device=device)


net = UNet(1, depth=3)

# Split training and validation data.
my_train_data = data[:-5].copy()
data_list = [data[i] for i in range(data.shape[0])]
my_val_data = data_list[-5:]
cfg_path = Path("./custom_config.yaml")
with open(cfg_path, "r") as stream:
    config_yml = yaml.safe_load(stream)
    config = Dict2ObjParser(config_yml).parse()
fine_tune = None
net = torch.load(path + "last_" + nameModel + ".net")

exp_path = Path("./exp")
Path.mkdir(exp_path, exist_ok=True)
pipeline = Pipeline.setup(config_yml)
pipeline(deepcopy(net), my_val_data, fine_tune, calibrate, exp_path)
pipeline.network2ir()
