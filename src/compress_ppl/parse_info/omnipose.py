import yaml
import torch
from cellpose import core, io, models, metrics
from src.utils import Dict2ObjParser
import os 

from .base import BaseParser

class OmniposeParser(BaseParser):
    """parse the omnipose model

    Args:
        Parser (_type_): base parser for the inherited parsers.
    """
    def __init__(self,config):
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
        use_gpu = self.meta_config.quantization.backend == 'tensorrt' and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        if self.args.pretrained_model != None and os.path.exists(self.args.pretrained_model):
            self.model = models.CellposeModel(gpu = use_gpu,
                                              pretrained_model = self.args.pretrained_model,
                                              device = self.device,
                                             )
            self.model.net.load_model(self.args.pretrained_model, 
                                    #   cpu=not self.args.use_gpu,
                                      device = self.device,
                                      )
        else:
            self.model = models.CellposeModel(gpu = use_gpu,
                                              model_type = self.args.model_type,
                                              # omni=self.args.omni,
                                              dim = self.args.dim
                                            )
        self.model.mkldnn = False #use openvino/tensorrt backend instead of mkldnn
        self.model.net.mkldnn = False
        return self.model
    
    def parse_data(self):
        files = io.get_image_files(self.args.data_path,mask_filter = '_masks')
        self.images = [io.imread(f) for f in files]
        # just for in house data:
        self.images = [img[:,:,0] for img in self.images] #only extract red channel
        return self.images
        
    def calibrate(self,model,calib_num: int): 
        """calibration step for the quantization to restore the precision.

        Args:
            model (_type_): model to be calibrated. Should be graph mode.
            calib_num (int): number of images to be used for calibration

        Raises:
            ValueError: _description_

        Returns:
            model: returned calibrated model
        """
        if calib_num <=0 or calib_num > len(self.images):
            raise ValueError('calibrate_num should be in range [1,{}]'.format(len(self.images)))
        model.net.to(self.device) #noted here Graphmodule object runs 1 time slower on gpu.
        model.eval( self.images[:calib_num], 
                    channels=self.args.channels,
                    diameter=self.args.diameter,
                    flow_threshold=self.args.flow_threshold,
                    cellprob_threshold=self.args.cellprob_threshold,
                    # omni=self.args.omni
                    )
        print(f'--------------calibration done!----------')
        return model


