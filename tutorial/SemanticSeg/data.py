import os
import numpy as np
from monai.transforms import (
    RandSpatialCropSamplesd,
    Compose,
    AddChanneld,
    ToTensord,
    Transform,
    CastToTyped,
    EnsureTyped,
    ScaleIntensityRangePercentilesd,
)
from tqdm.contrib import tenumerate
from aicsimageio import AICSImage


def generate_data_dict(data_path, gt_path):
    data_dicts = []

    for i, (data, label) in tenumerate(zip(os.listdir(data_path), os.listdir(gt_path))):
        data_dict = {}
        data_dict["img"] = os.path.join(data_path, data)
        data_dict["seg"] = os.path.join(gt_path, label)
        data_dict["fn"] = data.split(".")[0]
        data_dicts.append(data_dict)
    return data_dicts


class LoadTiffd(Transform):
    def __init__(self, keys=["img", "seg"]):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            x = AICSImage(data[key])
            d[key] = x.get_image_data("YX", S=0, T=0, C=0)
        return d


class Ins2Semd(Transform):
    def __init__(self, keys=["seg"]):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key][d[key] != 0] = 1
        return d


train_transform = Compose(
    [
        LoadTiffd(keys=["img", "seg"]),
        AddChanneld(keys=["img", "seg"]),
        CastToTyped(keys=["img"], dtype=np.float32),
        Ins2Semd(keys=["seg"]),
        EnsureTyped(keys=["img", "seg"]),
        ScaleIntensityRangePercentilesd(
            keys=["img"], lower=0.5, upper=99.5, b_min=0, b_max=1
        ),
        RandSpatialCropSamplesd(
            keys=["img", "seg"], roi_size=(256, 256), num_samples=4, random_size=False
        ),
        ToTensord(keys=["img", "seg"]),
    ]
)

test_transform = Compose(
    [
        LoadTiffd(keys=["img", "seg"]),
        AddChanneld(keys=["img", "seg"]),
        CastToTyped(keys=["img"], dtype=np.float32),
        Ins2Semd(keys=["seg"]),
        EnsureTyped(keys=["img", "seg"]),
        ScaleIntensityRangePercentilesd(
            keys=["img"], lower=0.5, upper=99.5, b_min=0, b_max=1
        ),
        ToTensord(keys=["img", "seg"]),
    ]
)
