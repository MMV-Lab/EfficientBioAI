#! /bin/bash 

# install forked MQBench to remove onnx requirement.
pip install git+https://github.com/audreyeternal/MQBench.git

# install development head of efficientbioai
cd EfficientBioAI
pip install -e .[$1]
cd ..

# install forked cellpose to fix some graph tracing problems.
pip install git+https://github.com/audreyeternal/cellpose.git

# install forked nni to extend to 3d tasks.
git clone https://github.com/audreyeternal/nni.git
cd nni
python setup.py develop 2>/dev/null
cd ..

# if $1=gpu/all, install torch 1.10 with cuda:
if [[ $1 == "gpu" ]] || [[ $1 == "all" ]]; then
    pip install --force-reinstall torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
fi

# make sure the numpy version is under 1.24 to satisfy mmv_im2im 0.4.0
pip install numpy==1.23