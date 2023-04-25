from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
]

requirements = [
    "numpy",
    "codecarbon",  # for measuring energy consumption
    "torch>=1.10, <1.11",
    "mmv_im2im",
    "cellpose",
    # "cellpose @ git+https://github.com/audreyeternal/cellpose.git", # pypi doesnt allow to install from git
    "nni",  # for pruning,
    "tensorboard",  # for visualization
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
]

opv_requirements = ["openvino", "openvino-dev[onnx,pytorch]"]

trt_requirements = [
    "pycuda",
    "tensorrt>=8.0, <8.6",
]

extra_requirements = {
    "cpu": opv_requirements,
    "gpu": trt_requirements,
    "all": [*opv_requirements, *trt_requirements],
    "test": test_requirements,
}

setup(
    name="efficientbioai",
    keywords="deep learning, quantization, microscopy model compression",
    description="efficientbioai is a python package for efficient deep learning in bioimaging",
    long_description=readme,
    long_description_content_type="text/markdown",
    version="0.0.6",
    author="mmv_lab team",
    author_email="yu.zhou@isas.de",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require=extra_requirements,  # install use: pip install efficientbioai[tensorrt/openvino/all]
    zip_safe=False,
)

# # only used when the environment is already set up
# setup(name = 'efficientbioai',
#       packages = find_packages(),
#       zip_safe=False)
