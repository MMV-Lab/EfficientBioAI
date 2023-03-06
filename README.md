# EfficientBioAI
This package mainly focus on the efficiency of BioImage AI tasks. For the moment we just implemented quantization algorithm.

## Introduction:
The whole project contains two parts: quantization and inference. In the quantization step, we quantize the pretrained model into int8 precision and transform them to the format suitable to the inference engine. The next step is to run the inference on the inference engine and do the analysis. The inference engine that we choose is `openvino` for intel CPU and `tensorrt` for nvidia GPU.   
We support several popular bioimage AI tools like([mmv_im2im](https://github.com/MMV-Lab/mmv_im2im),[cellpose](https://github.com/MouseLand/cellpose)). Also user-defined pytorch models are supported.
 

## Structure of the code:
```bash
├── configs
│   ├── config_base.yaml
│   ├── mmv_im2im
│   └── omnipose
├── data
│   ├── in_house_data
│   └── labelfree_3d
├── experiment
├── infer
│   ├── __init__.py
│   ├── openvino.py
│   ├── tensorrt.py
├── inference.py
├── model
│   ├── cellpose
│   ├── mmv_im2im
│   └── omnipose
├── onnx2trt.py
├── parse_info
│   ├── __init__.py
│   ├── mmv_im2im.py
│   ├── omnipose.py
├── quantization.py
└── utils.py
```
## Installation:
### pip:

### docker:(recommended)

## How to run it:
Take mmv_im2im for example:
- quantization:
 ```bash
python quantization.py --config configs/mmv_im2im/config_base.yaml --exp_path experiment/mmv_im2im
```
- inference and evaluation:
```bash
python inference.py --config experiment/mmv_im2im/mmv_im2im.yaml
```
- latency for one patch(normally [1,1,32,128,128]):
  - openvino:
    ```bash
    benchmark_app -m ./path/to/mmv_im2im_deploy_model.xml -nstream 1 -data_shape [1,1,32,128,128] -api sync
    ```
  - tensorrt: 

Note that pretrained model and data should be placed in the `model` and `data` folders, respectively. You can download our mmv_im2im pretrained model from [nextcloud](). 