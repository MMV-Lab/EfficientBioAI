from setuptools import setup, find_packages
import sys 


with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
]

requirements = [
    "numpy>=1.22, <1.24", # for some np.float syntax in mmv package that are deprecated since numpy 1.24
    "codecarbon",  # for measuring energy consumption
    "torch==1.10.0", # required by mqbench 
    "aicsimageio", # install it before mmv_im2im to avoid github action error
    "pytorch-lightning<=1.9.5", # should be less than 2.0 to satisfy mmv_im2im 0.4.0's requirement.
    "mmv_im2im==0.4.0", # fix to 0.4.0 at the moment. 
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
] if 'linux' in sys.platform else ["pycuda"]

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
