FROM python:3.8

RUN apt-get update && \
    apt-get install -y build-essential  && \
    apt-get install -y wget && \
    apt-get install -y libprotobuf-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN pip3 install --upgrade pip \
    && pip3 install \
    torch==1.10.0+cpu torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \  
    && pip3 install \
    openvino \
    openvino-dev[onnx,pytorch] \
    mmv_im2im \
    codecarbon \
    graphviz \
    cmake \
    onnx \
    nni

# install mqbench, cellpose
RUN git clone https://github.com/ModelTC/MQBench.git \
    && cd MQBench \
    && pip3 install -e . \
    && cd .. \
    && git clone https://github.com/audreyeternal/cellpose.git \
    && cd cellpose \
    && pip3 install -e .


