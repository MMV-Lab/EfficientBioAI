import os
import yaml
import shutil
import torch
from typing import Any, Callable, Union
from pathlib import Path

from .quantizer import Quantizer
from .pruner import Pruner
from efficientbioai.utils.misc import Dict2ObjParser
from efficientbioai.utils.logger import logger

_DEVICE = dict(openvino=torch.device("cpu"), tensorrt=torch.device("cuda"))


class Pipeline:
    """class for generating pipeline for model compression. Contains pruning and quantization."""

    def __init__(self, config_dict: dict, prune: bool = False, quantize: bool = True):
        self.config_dict = config_dict
        self.prune = prune
        self.quantize = quantize

    @classmethod
    def setup(cls, config_dict: dict):
        if (
            "prune" in config_dict.keys()
            and config_dict["quantization"]["run_mode"] == "int8"
        ):
            return cls(config_dict, True, True)
        elif (
            "prune" in config_dict.keys()
            and config_dict["quantization"]["run_mode"] == "fp32"
        ):
            return cls(config_dict, True, False)
        elif (
            "prune" not in config_dict.keys()
            and config_dict["quantization"]["run_mode"] == "int8"
        ):
            return cls(config_dict, False, True)
        elif (
            "prune" not in config_dict.keys()
            and config_dict["quantization"]["run_mode"] == "fp32"
        ):
            return cls(config_dict, False, False)
        else:
            raise NotImplementedError("compression strategy not supported!")

    def __call__(
        self,
        model: Any,
        data: Any,
        fine_tune: Callable,
        calibrate: Callable,
        path: Union[str, Path],
    ):
        self.config = Dict2ObjParser(self.config_dict).parse()
        self.input_size = self.config.data.input_size
        self.model_name = self.config.model.model_name
        self.config_path = self.config_dict["model"][self.model_name]["config_path"]
        self.input_names = self.config.data.io.input_names
        self.output_names = self.config.data.io.output_names
        self.output_path = path
        self.backend = self.config.quantization.backend
        self.device = _DEVICE[self.backend]
        self.run_mode = self.config.quantization.run_mode
        self.dynamic_batch = self.config.data.dynamic_batch
        logger.info(
            f"start to compress: quantize: {True if self.run_mode == 'int8' else False}, prune: {self.prune}, backend: {self.backend}, model_name: {self.model_name}"
        )
        # the prune and quantize process can only be done on cpu.
        if self.prune:
            pruner = Pruner(model, self.model_name, self.config_dict["prune"])
            model = pruner(self.input_size, data, fine_tune, device=torch.device("cpu"))
        quantizer = Quantizer(
            model,
            self.model_name,
            self.config_dict["quantization"],
            torch.device("cpu"),
        )
        quantizer(
            self.input_size,
            self.input_names,
            self.output_names,
            self.output_path,
            data,
            calibrate,
            fine_tune,
        )

    def network2ir(self):
        """from onnx to openvino ir or tensorrt engine.

        Raises:
            ImportError: pycuda not correctly installed
            ImportError: openvino mo module is not correctly installed or not in the path
            NotImplementedError: backend not supported
        """
        if self.backend == "tensorrt":
            try:
                from efficientbioai.onnx2trt import onnx2trt
            except Exception as e:
                raise ImportError("tensorrt/pycuda not correctly installed!") from e

            trt_path = os.path.join(self.output_path, f"{self.model_name}.trt")
            dynamic_file_path = os.path.join(
                self.output_path, f"{self.model_name}_clip_ranges.json"
            )
            onnx2trt(
                onnx_model=os.path.join(
                    self.output_path, f"{self.model_name}_deploy_model.onnx"
                ),
                trt_path=trt_path,
                mode=self.run_mode,
                dynamic_range_file=dynamic_file_path,
                input_names=self.input_names,
                input_size=self.input_size,
                dynamic_batch=self.dynamic_batch,
            )
            logger.info("transform done!")
            # save the config file to the folder:
            self.config_dict["model"][self.model_name]["model_path"] = trt_path
            self.config_dict["quantization"]["dynamic_range_file"] = dynamic_file_path
            self.config_dict["model"][self.model_name]["config_path"] = os.path.join(
                self.output_path, os.path.basename(self.config_path)
            )
            with open(
                os.path.join(self.output_path, f"{self.model_name}.yaml"), "w"
            ) as stream:
                yaml.dump(self.config_dict, stream)

        elif self.backend == "openvino":  # openvino
            import subprocess

            try:
                subprocess.run(
                    [
                        "mo",
                        "--input_model",
                        f"{self.output_path}/{self.model_name}_deploy_model.onnx",
                    ]
                )
            except Exception as e:
                raise ImportError(
                    "Please check openvino model optimizer installation path!"
                ) from e
            xml_path = os.path.join(
                self.output_path, f"{self.model_name}_deploy_model.xml"
            )
            self.config_dict["model"][self.model_name]["model_path"] = xml_path
            self.config_dict["model"][self.model_name]["config_path"] = os.path.join(
                self.output_path, os.path.basename(self.config_path)
            )
            with open(
                os.path.join(self.output_path, f"{self.model_name}.yaml"), "w"
            ) as stream:
                yaml.dump(self.config_dict, stream)
            for ext in ["bin", "mapping", "xml"]:
                try:
                    shutil.move(
                        os.path.join(os.getcwd(), f"{self.model_name}_deploy_model.{ext}"),
                        os.path.join(
                            self.output_path, f"{self.model_name}_deploy_model.{ext}"
                        ),
                    )
                except: # sometimes there is no mapping file.
                    continue
            logger.info("transform done!")

        else:
            raise NotImplementedError("backend not supported!")
