#!/usr/bin/env python3
import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from unet.model import UNet
import torch
from pn2v import utils
from pn2v import training
from tifffile import imread
from scipy.ndimage import gaussian_filter
import glob
import random

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", help="name of your network", default='N2V')
parser.add_argument("--dataPath", help="path to your training data and where network will be stored")
parser.add_argument("--fileName", help="name of your training data file", default='*.tif')
parser.add_argument("--validationFraction", help="Fraction of data you want to use for validation (percent)", default=30.0, type=float)
parser.add_argument("--patchSizeXY", help="XY-size of your training patches", default=100, type=int)
parser.add_argument("--epochs", help="number of training epochs", default=200, type=int)
parser.add_argument("--stepsPerEpoch", help="number training steps per epoch", default=10, type=int)
parser.add_argument("--batchSize", help="size of your training batches", default=1, type=int)
parser.add_argument("--virtualBatchSize", help="size of virtual batch", default=20, type=int)
parser.add_argument("--netDepth", help="depth of your U-Net", default=3, type=int)
parser.add_argument("--learningRate", help="initial learning rate", default=1e-3, type=float)
parser.add_argument("--netKernelSize", help="size of conv. kernels in first layer", default=3, type=int)
parser.add_argument("--unet_n_first", help="number of feature channels in the first u-net layer", default=64, type=int)
parser.add_argument("--sizePSF", help="size of psf in pix, odd number", default=81, type=int)
parser.add_argument("--stdPSF", help="size of std of gauss for psf", default=1.0, type=float)
parser.add_argument("--positivityConstraint", help="positivity constraint parameter", default=1.0, type=float)
parser.add_argument("--meanValue", help="mean value for the background ", default=0.0, type=float)

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
print(args)

# See if we can use a GPU
device = utils.getDevice()

print("args",str(args.name))

####################################################
#           PREPARE TRAINING DATA
####################################################
path = args.dataPath
files = sorted(glob.glob(path + args.fileName))
files.sort(key = lambda s: len(s))
# Load the training data
data = []
for f in files:
    current_img = imread(f)
    data.append(current_img.astype(np.float32))

data = np.array(data) - args.meanValue

if len(data.shape)==4:
    data.shape = (data.shape[0]*data.shape[1],data.shape[2],data.shape[3])

####################################################
#           PREPARE PSF
####################################################

def artificial_psf(size_of_psf = args.sizePSF, std_gauss = args.stdPSF):  
    filt = np.zeros((size_of_psf, size_of_psf))
    p = (size_of_psf - 1)//2
    filt[p,p] = 1
    filt = torch.tensor(gaussian_filter(filt,std_gauss).reshape(1,1,size_of_psf,size_of_psf).astype(np.float32))
    filt = filt/torch.sum(filt)
    return filt
psf_tensor = artificial_psf()

####################################################
#           CREATE AND TRAIN NETWORK
####################################################
net = UNet(1, depth=args.netDepth)
net.psf = psf_tensor.to(device)
# Split training and validation data
splitter = np.int(data.shape[0] * args.validationFraction/100.)
print("splitter = ", splitter)
my_train_data = data[:-splitter].copy()
my_val_data = data[-splitter:].copy()

# Start training
trainHist, valHist = training.trainNetwork(net = net, trainData = my_train_data, valData = my_val_data,
                                           postfix = args.name, directory = path, noiseModel = None,
                                           device = device, numOfEpochs = args.epochs, patchSize = args.patchSizeXY, stepsPerEpoch = 10,
					   virtualBatchSize = 20, batchSize = args.batchSize, learningRate = 1e-3,psf = psf_tensor.to(device), 
					   positivity_constraint = args.positivityConstraint)
