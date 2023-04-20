# EfficientBioAI
This package mainly focus on the efficiency of BioImage AI tasks. For the moment we just implemented quantization algorithm.

## Introduction:
The whole project contains two parts: quantization and inference. In the quantization step, we quantize the pretrained model into int8 precision and transform them to the format suitable to the inference engine. The next step is to run the inference on the inference engine and do the analysis. The inference engine that we choose is `openvino` for intel CPU and `tensorrt` for nvidia GPU.   
We support several popular bioimage AI tools like([mmv_im2im](https://github.com/MMV-Lab/mmv_im2im),[cellpose](https://github.com/MouseLand/cellpose)). Also user-defined pytorch models are supported.
 
## Installation:
### pip:
First create a virtual environment using conda:
```bash
conda config --add channels conda-forge
conda create -n efficientbioai python=3.8 setuptools=59.5.0
```
Then we need to install the dependencies:
```bash
git clone git@github.com:ModelTC/MQBench.git
cd MQBench
python setup.py install
cd ..
```
Then install the `efficientbioai` package:

```bash
git clone git@github.com:MMV-Lab/EfficientBioAI.git
cd EfficientBioAI
pip install -e .[cpu/gpu/all] # for intel cpu, nvidia gpu or both
```

### docker:(recommended)
We use different docker images for both cpu and gpu. Assume that you are in the root directory of the project.
- for CPU:
```bash
cd docker/cpu
bash install.sh # if not install docker, run this command first
bash build_docker.sh # build the docker image
cd ../..
bash docker/cpu/run_container.sh #run the docker container
```
- for GPU:
```bash
cd docker/gpu
bash install.sh # if not install docker, run this command first
bash build_docker.sh # build the docker image
cd ../..
bash docker/gpu/run_container.sh #run the docker container
```

## How to run it:
### Use scripts:
- compression:
 ```bash
python efficientbioai/compress.py --config path/to/the/config.yaml --exp_path experiment/save_path
```
- inference:
```bash
python efficientbioai/inference.py --config path/to/the/config.yaml
```
### Use functions:
There is a simple [example](tutorial/compress_custom_network.ipynb)


Note that pretrained model and data should be placed in the `model` and `data` folders, respectively. You can download our mmv_im2im pretrained model from [nextcloud](). All the intermediate files will be saved in the `experiment` folder. 