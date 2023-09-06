import torch
import os
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset
from cellpose import io, models, metrics

class Omnipose():

    def __init__(self, configure):
        super().__init__()
        self.args = configure
        self.model = self.parse_model()
        print(type(self.model))
        self.images, self.masks, self.files = io.load_images_labels(self.args.data_path)
        
    
    def parse_model(self, device=torch.device("cpu")):
        """parse cellpose/omnipose model. read the pretrained model if it exists.

        Returns:
            model: _description_
        """

        if self.args.pretrained_model is not None and os.path.exists(
            self.args.pretrained_model
        ):
            self.model = models.CellposeModel(
                gpu=False,
                pretrained_model=self.args.pretrained_model,
                device=device,
            )
            self.model.net.load_model(
                self.args.pretrained_model,
                #   cpu=not self.args.use_gpu,
                device=device,
            )
        else:
            self.model = models.CellposeModel(
                gpu=False,
                model_type=self.args.model_type,
                # omni=self.args.omni,
                dim=self.args.dim,
            )
        self.model.mkldnn = False  # use openvino/tensorrt backend instead of mkldnn
        self.model.net.mkldnn = False
        return self.model
    
    def get_data(self):
        return self.images, self.masks 

    def get_model(self):
        return self.model

    def core_infer(self,images):
        self.model.net.eval()
        # torch.onnx.export(self.model.net,torch.randn(1,2,224,224),"./cellpose.onnx")
        output = self.model.net(images)
        return output

    def infer(self):
        masks, flows, _ = self.model.eval(
            self.images,
            channels=self.args.channels,
            diameter=self.args.diameter,
            flow_threshold=self.args.flow_threshold,
            cellprob_threshold=self.args.cellprob_threshold,
        )
        self.pred_masks = masks
        self.pred_flows = flows

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