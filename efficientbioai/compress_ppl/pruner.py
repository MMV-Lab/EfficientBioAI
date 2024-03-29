import os
from typing import Optional, Union, Callable, List, Any
import torch
import logging

logging.getLogger("nni").setLevel(logging.ERROR)
from nni.runtime.log import silence_stdout, _root_logger  # noqa E402

silence_stdout()
_root_logger.handlers = []

from nni.compression.pytorch import pruning  # noqa E402
from nni.compression.pytorch.speedup import ModelSpeedup  # noqa E402

from efficientbioai.utils.logger import logger  # noqa E402


class Pruner:
    """class for pruning algorithm."""

    def __init__(self, model: Any, model_type: str, pconfig: dict):
        self.model = model
        self.model_type = model_type
        self.pconfig = pconfig

    def _get_network(self):
        """extract the part in the torch module for quantization."""
        if self.model_type in ["mmv_im2im", "cellpose", "omnipose"]:
            self.network = self.model.net
        elif self.model_type == "academic":
            self.network = self.model
        else:
            raise NotImplementedError("model type not supported!")

    def _set_network(self):
        """retrive the quantized network and insert back to the original model."""
        if self.model_type in ["mmv_im2im", "cellpose", "omnipose"]:
            self.model.net = self.network
        elif self.model_type == "academic":
            self.model = self.network
        else:
            raise NotImplementedError("model type not supported!")

    def __call__(
        self,
        input_size: List[int],  # [C, H, W]
        data: Any,
        fine_tune: Callable,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    ) -> Any:
        logger.info("start pruning with type:{}...".format(self.pconfig["type"]))
        if not isinstance(input_size, list):
            logger.error("Input 'input_size' should be a list")
            raise TypeError("Input 'input_size' should be a list")
        # elif len(input_size) > 3:
        #     error_message = " Currently Pruning is not supported for dimensions higher than 2D. Given input_size: {}".format(
        #         input_size
        #     )
        #     logger.warning(error_message)
        #     raise ValueError(error_message)
        self._get_network()
        dummy_input = torch.rand(1, *input_size).to(device)
        Pruner = getattr(pruning, self.pconfig["type"])
        pruner = Pruner(
            self.network,
            self.pconfig["config_list"],
            mode="dependency_aware",
            dummy_input=dummy_input,
        )
        pruner._unwrap_model()
        # compress the model and generate the masks
        _, masks = pruner.compress()
        # show the masks sparsity
        for name, mask in masks.items():
            print(
                name,
                " sparsity : ",
                "{:.2}".format(mask["weight"].sum() / mask["weight"].numel()),
                " shape : ",
                mask["weight"].shape,
            )
        ModelSpeedup(
            self.network,
            torch.rand(1, *input_size).to(device),
            masks,
            customized_replace_func=self.pconfig["customized_replace_func"],
        ).speedup_model()
        self._set_network()
        logger.info("start fine-tuning to improve performance...")
        fine_tune(self.model, data, device)
        logger.info("pruning finished!")
        return self.model

    @staticmethod
    def export_network(
        model, model_type, input_size, input_names, output_names, output_path
    ):
        io_names = [*input_names, *output_names]
        dynamic_axes = {k: {0: "batch_size"} for k in io_names}
        torch.onnx.export(
            model,
            torch.randn(1, *input_size),
            os.path.join(output_path, f"{model_type}_deploy_model.onnx"),
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
