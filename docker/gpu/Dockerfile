# replace the tensorrt version based on your nvidia driver version. https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-09.html#rel_21-09
ARG version=22.09-py3
FROM nvcr.io/nvidia/tensorrt:${version}

# install packages, cuda support max to 11.3
RUN pip3 install --upgrade pip \
    && pip3 install \
    torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html \  
    pytorch-lightning==1.8.1 \
    mmv_im2im \
    codecarbon \
    graphviz \
    nni

# install mqbench, cellpose
RUN git clone https://github.com/ModelTC/MQBench.git \
    && cd MQBench \
    && python setup.py install \
    && cd ../ \
    && git clone https://github.com/audreyeternal/cellpose.git \
    && cd cellpose \
    && pip install -e .

# WORKDIR /EfficientBioAI
