import time
from pathlib import Path
from typing import Sequence
from importlib import import_module

import torch
import numpy as np
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from skimage.io import imsave as save_rgb
from torchmetrics import Dice, StructuralSimilarityIndexMeasure, PearsonCorrCoef
from monai.inferers import sliding_window_inference
from tqdm.contrib import tenumerate
from mmv_im2im.utils.misc import generate_test_dataset_dict, parse_config
from mmv_im2im.utils.for_transform import parse_monai_ops_vanilla
from typing import Dict, List, Sequence, Text, Type, Union, TypeVar, Generic, Optional


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class Mmv():
    """class for Mmv_im2im model"""

    def __init__(self, mmv_args) -> None:  # define the model
        self.args = mmv_args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data_cfg = self.args.data
        self.model_cfg = self.args.model
        self.model = self.parse_model()
        

    def parse_model(self, device=torch.device("cpu")):
        model_category = self.model_cfg.framework
        model_module = import_module(f"mmv_im2im.models.pl_{model_category}")
        my_model_func = getattr(model_module, "Model")
        mmv_model = my_model_func(self.model_cfg, train=False)
        # pre_train = torch.load(self.model_cfg.checkpoint)
        # mmv_model.load_state_dict(pre_train["state_dict"])
        # torch.onnx.export(mmv_model.net,torch.randn(1,1,128,128),"./mmv.onnx")
        mmv_model.to(self.device)
        return mmv_model

    def get_model(self):
        return self.model
    
    def prepare_data(self):
        self.dataset_list = generate_test_dataset_dict(
            self.data_cfg.inference_input.dir, self.data_cfg.inference_input.data_type
        )

        self.dataset_length = len(self.dataset_list)
        if "Z" in self.data_cfg.inference_input.reader_params["dimension_order_out"]:
            self.spatial_dims = 3
        else:
            self.spatial_dims = 2

        if self.data_cfg.preprocess is not None:
            # load preprocessing transformation
            self.pre_process = parse_monai_ops_vanilla(self.data_cfg.preprocess)

    def save_result(self, pred: np.ndarray, out_fn) -> None:
        if out_fn.suffix == ".npy":
            np.save(out_fn, pred)
        else:
            if len(pred.shape) == 2:
                OmeTiffWriter.save(pred, out_fn, dim_order="YX")
            elif len(pred.shape) == 3:
                # 3D output, for 2D data
                if self.spatial_dims == 2:
                    # save as RGB or multi-channel 2D
                    if pred.shape[0] == 3:
                        if out_fn.suffix != ".png":
                            out_fn = out_fn.with_suffix(".png")
                        save_rgb(out_fn, np.moveaxis(pred, 0, -1))
                    else:
                        OmeTiffWriter.save(pred, out_fn, dim_order="CYX")
                elif self.spatial_dims == 3:
                    OmeTiffWriter.save(pred, out_fn, dim_order="ZYX")
                else:
                    raise ValueError("Invalid spatial dimension of pred")
            elif len(pred.shape) == 4:
                if self.spatial_dims == 3:
                    OmeTiffWriter.save(pred, out_fn, dim_order="CZYX")
                elif self.spatial_dims == 2:
                    if pred.shape[0] == 1:
                        if pred.shape[1] == 1:
                            OmeTiffWriter.save(pred[0, 0], out_fn, dim_order="YX")
                        elif pred.shape[1] == 3:
                            if out_fn.suffix != ".png":
                                out_fn = out_fn.with_suffix(".png")
                            save_rgb(
                                out_fn,
                                np.moveaxis(
                                    pred[0,],  # noqa: E231
                                    0,
                                    -1,
                                ),
                            )
                        else:
                            OmeTiffWriter.save(
                                pred[0,],  # noqa: E231
                                out_fn,
                                dim_order="CYX",
                            )
                    else:
                        raise ValueError("invalid 4D output for 2d data")
            elif len(pred.shape) == 5:
                assert pred.shape[0] == 1, "error, found non-trivial batch dimension"
                OmeTiffWriter.save(
                    pred[0,],  # noqa: E231
                    out_fn,
                    dim_order="CZYX",
                )
            else:
                raise ValueError("error in prediction output shape")

    def process_one_image(self, ds):
        img = AICSImage(ds).reader.get_image_dask_data(
            **self.data_cfg.inference_input.reader_params
        )
        x = img.compute()
        x = torch.tensor(x.astype(np.float32))
        x = self.pre_process(x)  # normalize to [0,1]
        x = x.as_tensor().unsqueeze(0)  # bczyx
        del img
        return x

    
    def core_infer(self,images):
        """core infer function of mmv package.

        Args:
            images (torch.Tensor): input, should be BC(Z)YX
        """
        output = self.model.net(images)
        return output


    def infer(self):
        self.prepare_data()
        use_window_inference = True
        infer_time = AverageMeter()
        with torch.no_grad():
            for i, ds in tenumerate(self.dataset_list):
                fn_core = Path(ds).stem
                # suffix = self.data_cfg.inference_output.suffix
                img = AICSImage(ds).reader.get_image_dask_data(
                    **self.data_cfg.inference_input.reader_params
                )
                x = img.compute()
                x = torch.tensor(x.astype(np.float32))
                if self.pre_process is not None:
                    x = self.pre_process(x)
                x = x.unsqueeze(0).unsqueeze(0).as_tensor().to(self.device)
                # calcuate avg time for the whole image
                end = time.time()
                if (
                    self.model_cfg.model_extra is not None
                    and "sliding_window_params" in self.model_cfg.model_extra
                    and use_window_inference
                ):
                    y_hat = sliding_window_inference(
                        inputs=x,
                        predictor=self.model,
                        device=torch.device("cpu"),
                        **self.model_cfg.model_extra["sliding_window_params"],
                    )
                else:
                    y_hat = self.model(x)
                latency = time.time() - end
                print(f"latency for {fn_core} is {latency:.3f}")
                infer_time.update(latency)
                if self.data_cfg.postprocess is not None:
                    pp_data = y_hat
                    for pp_info in self.data_cfg.postprocess:
                        pp = parse_config(pp_info)
                        pp_data = pp(pp_data)
                    if torch.is_tensor(pp_data):
                        pred = pp_data.cpu().numpy()
                    else:
                        pred = pp_data
                else:
                    pred = y_hat.cpu().numpy()
                out_fn = (
                    Path(self.data_cfg.inference_output.path)
                    / f"{fn_core}.tif"  # need to add _{suffix} if input and output path are different.
                )
                self.save_result(pred, out_fn)
        avg_infer_time = infer_time.avg
        print(f"average inference time is {avg_infer_time:.3f}")

    def evaluate(
        self,
        metric: Sequence[str],
    ) -> None:
        """evaluation for mmv_im2im related tasks. Need to specify data location and data type.

        Args:
            metric (Sequence[str]): evaluation metrics.Currently support:
                1. SSIM for labelfree transformation
                2. Dice for semantic segmentation
                3. Pearson correlation for labelfree transformation
        """

        metric_table = dict(
            SSIM=StructuralSimilarityIndexMeasure(),
            Dice=Dice(average="micro", ignore_index=0),
            Pearson=PearsonCorrCoef(),
        )
        our_metric = {}
        for k in metric:
            try:
                our_metric[k] = metric_table[k].to(
                    torch.device("cpu")
                )  # probably out of memory if use gpu
            except Exception as e:
                raise ValueError(f"metric {k} not supported") from e
        # read gt/pred file in order. Suppose the file names are the same.
        gt_dir = self.data_cfg.inference_output.path
        gt_data_type = self.data_cfg.inference_input.data_type
        gt_list = generate_test_dataset_dict(gt_dir, gt_data_type)
        pred_list = self.dataset_list
        if self.data_cfg.preprocess is not None:
            # load preprocessing transformation
            self.pre_process = parse_monai_ops_vanilla(self.data_cfg.preprocess)

        for i, (gt, pred) in tenumerate(zip(gt_list, pred_list)):
            gt = self.process_one_image(gt)
            pred = self.process_one_image(pred)
            if "Dice" in metric:  # for semantic_seg
                act_layer = torch.nn.Softmax(dim=1)
                yhat_act = act_layer(pred).numpy()
                out_img = np.argmax(yhat_act, axis=1, keepdims=True).astype(np.uint8)
                out_img = torch.from_numpy(out_img)
                our_metric["Dice"].update(gt, out_img)
            if "SSIM" in metric:  # for domain adaptation
                our_metric["SSIM"].update(gt, pred)
            if "Pearson" in metric:  # for domain adaptation
                gt = torch.flatten(gt.squeeze(0).squeeze(0))
                pred = torch.flatten(pred.squeeze(0).squeeze(0))
                our_metric["Pearson"].update(gt, pred)

        metric_summary = {}
        for k, v in our_metric.items():
            score = v.compute()
            print(k + f" score is {score:.3f}")
            metric_summary[k] = score
        return metric_summary