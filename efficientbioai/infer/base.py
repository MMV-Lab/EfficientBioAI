from abc import ABC, abstractmethod
import os
import time

import torch
from codecarbon import EmissionsTracker

# from .backend import create_opv_model, create_trt_model
from efficientbioai.parse_info import Mmv_im2imParser, OmniposeParser
from efficientbioai.utils import Dict2ObjParser, AverageMeter

_DEVICE = dict(openvino=torch.device("cpu"), tensorrt=torch.device("cuda"))
_PARSER = dict(omnipose=OmniposeParser, mmv_im2im=Mmv_im2imParser)


def check_device(backend):
    if not torch.cuda.is_available() and backend == "tensorrt":
        raise ValueError("TensorRT backend requires CUDA to be available")
    else:
        print("Using {} backend, device checked!".format(backend))


def create_model(backend, model_path):
    if backend == "openvino":
        from .backend.openvino import create_opv_model

        return create_opv_model(model_path)
    elif backend == "tensorrt":
        from .backend.tensorrt import create_trt_model

        return create_trt_model(model_path)
    else:
        raise ValueError("backend {} is not supported".format(backend))


class BaseInfer:
    def __init__(self, config_yml: dict) -> None:
        """
        Initialize the base class for inference.
            1. Parse the config file
            2. Define and check the device
            3. Create the inference network
            4. Create the parser
        """
        configure = Dict2ObjParser(config_yml).parse()
        self.backend = config_yml["quantization"]["backend"]
        check_device(self.backend)
        self.device = _DEVICE[self.backend]
        self.model_name = config_yml["model"]["model_name"]
        cfg_path = config_yml["model"][self.model_name]["config_path"]
        self.base_path = os.path.split(cfg_path)[0]
        infer_path = config_yml["model"][self.model_name]["model_path"]
        self.parser = _PARSER[self.model_name](configure)
        self.network = create_model(self.backend, infer_path)
        self.config = self.parser.config
        self.input_size = configure.data.input_size

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def run_infer(self):
        pass

    @abstractmethod
    def save_result(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def calculate_infer_time(self, num: int = 1000) -> None:
        """calculating inference time using only patches, not the whole image. circulate num times, take the average.

        Args:
            num (int): number of patches to be inferenced.
        """
        infer_time = AverageMeter()
        infer_data = [
            torch.randn(1, *self.input_size, device=self.device) for _ in range(num)
        ]
        for x in infer_data:
            end = time.time()
            self.network(x)
            infer_time.update(time.time() - end)
        avg_infer_time = infer_time.avg
        print(f"average inference time is {avg_infer_time:.3f}")

    @abstractmethod
    def calculate_energy(self, num: int = 1000) -> float:
        """calculate energy consumption using only patches, not the whole image. circulate num times, take the average. The value is based on codecarbon package.

        Args:
            num (int): number of patches to be inferenced.

        Returns:
            float: carbon dioxide emission in grams
        """
        infer_data = [
            torch.randn(1, *self.input_size, device=self.device) for _ in range(num)
        ]
        # self.model.net.to(self.device)
        tracker = EmissionsTracker(measure_power_secs=1, output_dir=self.base_path)
        tracker.start()
        with torch.no_grad():
            for x in infer_data:
                self.network(x)
        emissions: float = tracker.stop()
        print(emissions)
