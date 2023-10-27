FROM python:3.8

RUN apt-get update && \
    apt-get install -y build-essential  && \
    apt-get install -y wget && \
    apt-get install -y libprotobuf-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install cmake, required by onnx.
RUN pip3 install --upgrade pip \
    && pip3 install \
    cmake

RUN git clone https://github.com/MMV-Lab/EfficientBioAI.git \
    && chmod 777 ./EfficientBioAI/installation/setup.sh \
    && ./EfficientBioAI/installation/setup.sh cpu

RUN pip3 install --upgrade pip \
    && pip3 install \
    torch==1.10.0+cpu torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    numpy==1.23


