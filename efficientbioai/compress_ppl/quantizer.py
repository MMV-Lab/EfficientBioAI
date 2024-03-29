import os
from typing import Sequence, Any, Union, List, Callable
from pathlib import Path
import warnings

import torch
from torch.utils.data import DataLoader
from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import BackendType, prepare_by_platform
from mqbench.utils.state import enable_calibration, enable_quantization

from efficientbioai.utils.logger import logger

warnings.filterwarnings("ignore")

_BACKEND = dict(
    tensorrt=BackendType.Tensorrt,
    openvino=BackendType.OPENVINO,
)


class Quantizer:
    """Quantizer class for quantizing a model."""

    def __init__(self, model: Any, model_type: str, qconfig: dict = None, device=None):
        self.model = model
        self.model_type = model_type
        self.qconfig = qconfig
        self.device = device

    def _get_network(self):
        """extract the part in the torch module for quantization."""
        if self.model_type in ["mmv_im2im", "cellpose", "omnipose"]:
            self.network = self.model.net
        elif self.model_type == "academic":  # for the custom model
            self.network = self.model
        else:
            err_msg = "model type not supported!"
            logger.error(err_msg)
            raise NotImplementedError(err_msg)

    def _set_network(self):
        """retrive the quantized network and insert back to the original model."""
        if self.model_type in ["mmv_im2im", "cellpose", "omnipose"]:
            self.model.net = self.network
        elif self.model_type == "academic":
            self.model = self.network
        else:
            err_msg = "model type not supported!"
            logger.error(err_msg)
            raise NotImplementedError(err_msg)

    def _quantize(
        self,
        data: Any,
        fine_tune: Callable = None,
        type: str = "PTQ",
    ):
        self._get_network()  # get the network to be quantized
        if type.upper() == "PTQ":
            enable_quantization(
                self.network
            )  # turn on actually quantization, ready for simulating Backend inference
        elif type.upper() == "QAT":
            assert (
                fine_tune is not None
            ), "fine_tune function should be provided for QAT!"
            self.network.train()
            self._set_network()
            fine_tune(self.model, data, device=self.device)
            self.network.eval()
            self._get_network()
            enable_quantization(self.network)
        else:
            err_msg = "quantization type not supported!"
            logger.error(err_msg)
            raise NotImplementedError(err_msg)

    def __call__(
        self,
        input_size: List[int],
        input_names: List[str],
        output_names: List[str],
        output_path: Union[str, Path],
        data: Union[DataLoader, Sequence[Any], None] = None,
        calibrate: Callable = None,
        fine_tune: Callable = None,
    ):
        """Quantize step implementation using MQBench api.

        Args:
            input_size (List[int]): input size
            input_names (List[str]): input names
            output_names (List[str]): output names
            output_path (Union[str, Path]): path for saving the quantized onnx model
            data (Union[DataLoader, Sequence[Any], None], optional): data for calibration. Defaults to None.
            calibrate (_type_, optional): calibration step, defined in the Parser class. Defaults to None.
        """

        logger.info("Start quantization...")
        input_shape = {
            input_names[0]: [1, *input_size]
        }  # batchsize+channel, ZYX. only consider 1 input senario.
        io_names = [*input_names, *output_names]
        dynamic_axes = {k: {0: "batch_size"} for k in io_names}
        backend = _BACKEND[self.qconfig["backend"]]
        self._get_network()
        if self.qconfig["run_mode"] == "int8":
            # since additional model type can only accept tuple. so we need to convert it to tuple.
            extra_config = {
                "extra_qconfig_dict": self.qconfig["extra_config"][
                    "extra_qconfig_dict"
                ],
                "extra_quantizer_dict": {
                    "additional_module_type": (
                        torch.nn.Conv3d,
                        torch.nn.MaxPool3d,
                        torch.nn.ConvTranspose3d,
                    ),
                    "additional_function_type": [
                        torch.cat,
                    ],
                },
            }
            self.network = prepare_by_platform(
                self.network, backend, extra_config
            )  # trace model and add quant nodes for model on backend
            enable_calibration(
                self.network
            )  # turn on calibration, ready for gathering data
            self._set_network()
            self.network = calibrate(
                self.model, data, device=self.device
            )  # run calibration
            self._quantize(data, fine_tune=fine_tune, type=self.qconfig["type"])
            extra_kwargs = dict(
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
            convert_deploy(
                self.network,
                backend,
                input_shape,
                model_name=self.model_type,
                output_path=output_path,
                deploy_to_qlinear=False,
                **extra_kwargs,
            )
        else:  # run in fp32 mode.
            logger.info("Running in fp32 mode...")
            self.network.eval()
            with torch.no_grad():
                torch.onnx.export(
                    self.network,
                    torch.randn(1, *input_size),
                    os.path.join(output_path, f"{self.model_type}_deploy_model.onnx"),
                    verbose=True,
                    opset_version=14,
                    input_names=input_names,
                    output_names=output_names,
                    do_constant_folding=True,
                    dynamic_axes=dynamic_axes,
                )
        logger.info("Quantization finished!")
