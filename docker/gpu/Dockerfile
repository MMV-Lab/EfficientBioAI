# replace the tensorrt version based on your nvidia driver version. https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-09.html#rel_21-09
ARG version=22.09-py3
FROM nvcr.io/nvidia/tensorrt:${version}

RUN git clone https://github.com/MMV-Lab/EfficientBioAI.git \
    && chmod 777 ./EfficientBioAI/installation/setup.sh \
    && ./EfficientBioAI/installation/setup.sh all

# install packages, cuda support max to 11.3
RUN pip3 install --upgrade pip \
    && pip3 install \
    torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html \   
    numpy==1.23
