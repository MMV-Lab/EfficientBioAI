#! /bin/bash 

# install mqbench because it cannot be installed from pypi
pip install git+https://github.com/ModelTC/MQBench.git

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

