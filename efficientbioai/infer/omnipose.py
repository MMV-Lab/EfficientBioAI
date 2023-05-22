import numpy as np
from cellpose import io, metrics
from codecarbon import track_emissions
from efficientbioai.utils.misc import timer

from .base import BaseInfer


class OmniposeInfer(BaseInfer):
    """inference for omnipose model."""

    def __init__(self, config_yml) -> None:  # define the model
        super().__init__(config_yml)
        model = self.parser.parse_model(self.device)
        model.mkldnn = False
        model.net = self.network
        self.model = model
        self.data_dir = self.config.data_path
        self.images, self.masks, self.files = None, None, None

    def prepare_data_without_gt(self):
        img_names = io.get_image_files(
            self.data_dir, mask_filter=self.config.mask_filter
        )
        self.images = []
        for img_name in img_names:
            img = io.imread(img_name)
            img = img[
                :, :, 0
            ]  # only use the red channel, only for in house data in 2d instance segmentation task
            self.images.append(img)

    def prepare_data(self):
        output = io.load_images_labels(
            tdir=self.data_dir,
            image_filter=self.config.image_filter,
            mask_filter=self.config.mask_filter,
        )

        self.images, self.masks, self.files = output
        # just for in house data(created by shuo):
        # because the data is RGB, we only extract the red channel,
        # the mask type is float32, we convert it to uint16
        self.images = [img[:, :, 0] for img in self.images]
        self.masks = [f.astype(np.uint16) for f in self.masks]

    @track_emissions(measure_power_secs=1)
    @timer
    def core_infer(self):
        masks, flows, _ = self.model.eval(
            self.images,
            channels=self.config.channels,
            diameter=self.config.diameter,
            flow_threshold=self.config.flow_threshold,
            cellprob_threshold=self.config.cellprob_threshold,
        )
        self.pred_masks = masks
        self.pred_flows = flows

    def save_result(self):
        io.save_masks(
            self.images,
            self.pred_masks,
            self.pred_flows,
            self.files,
            savedir=self.base_path,
            save_txt=False,  # save txt outlines for ImageJ
            save_flows=False,  # save flows as TIFFs
            tif=True,
        )

    def run_infer(self):
        # self.prepare_data()
        # self.core_infer()
        # self.save_result()
        self.prepare_data_without_gt()
        self.core_infer()

    def evaluate(self):
        if self.masks is not None:  # if gt masks are provided, compute AP
            threshold = [0.5, 0.75, 0.9]
            ap, tp, fp, fn = metrics.average_precision(
                self.masks, self.pred_masks, threshold=threshold
            )
            print(
                f"AP50 is {sum(ap[:,0])/len(ap[:,0])}, AP75 is {sum(ap[:,1])/len(ap[:,1])}, AP90 is {sum(ap[:,2])/len(ap[:,2])}"
            )
        else:
            print("no ground truth masks provided, skipping evaluation!")
