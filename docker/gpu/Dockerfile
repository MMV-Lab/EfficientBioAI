# replace the tensorrt version based on your nvidia driver version. https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-09.html#rel_21-09
ARG version=22.09-py3
FROM nvcr.io/nvidia/tensorrt:${version}

# install packages, cuda support max to 11.3
RUN pip3 install --upgrade pip \
    && pip3 install \
    torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html \  
    && pip3 install \
    mmv_im2im \
    codecarbon \
    graphviz

# comment a bug in pytorch

WORKDIR /tensorrt_docker
# install mqbench
RUN git clone https://github.com/ModelTC/MQBench.git \
    && cd MQBench \
    && python setup.py install

# install cellpose
RUN git clone https://github.com/MouseLand/cellpose.git \
    && cd cellpose \
    && pip install -e .
