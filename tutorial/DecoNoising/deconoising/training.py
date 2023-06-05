import torch.optim as optim
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from pn2v import utils


############################################
#   Training the network
############################################


def getStratifiedCoords2D(numPix, shape):
    '''
    Produce a list of approx. 'numPix' random coordinate, sampled from 'shape' using startified sampling.
    '''
    box_size = np.round(np.sqrt(shape[0] * shape[1] / numPix)).astype(np.int)
    coords = []
    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    for i in range(box_count_y):
        for j in range(box_count_x):
            y = np.random.randint(0, box_size)
            x = np.random.randint(0, box_size)
            y = int(i * box_size + y)
            x = int(j * box_size + x)
            if (y < shape[0] and x < shape[1]):
                coords.append((y, x))
    return coords


def randomCropFRI(data, size, numPix, supervised=False, counter=None, augment=True):
    '''
    Crop a patch from the next image in the dataset.
    The patches are augmented by randomly deciding to mirror them and/or rotating them by multiples of 90 degrees.
    
    Parameters
    ----------
    data: numpy array
        your dataset, should be a stack of 2D images, i.e. a 3D numpy array
    size: int
        witdth and height of the patch
    numPix: int
        The number of pixels that is to be manipulated/masked N2V style.
    dataClean(optinal): numpy array 
        This dataset could hold your target data e.g. clean images.
        If it is not provided the function will use the image from 'data' N2V style
    counter (optinal): int
        the index of the next image to be used. 
        If not set, a random image will be used.
    augment: bool
        should the patches be randomy flipped and rotated?
    
    Returns
    ----------
    imgOut: numpy array 
        Cropped patch from training data
    imgOutC: numpy array
        Cropped target patch. If dataClean was provided it is used as source.
        Otherwise its generated N2V style from the training set
    mask: numpy array
        An image holding marking which pixels should be used to calculate gradients (value 1) and which not (value 0)
    counter: int
        The updated counter parameter, it is increased by one.
        When the counter reaches the end of the dataset, it is reset to zero and the dataset is shuffled.
    '''
    
    if counter is None:
        index=np.random.randint(0, data.shape[0])
    else:
        if counter>=data.shape[0]:
            counter=0
            np.random.shuffle(data)
        index=counter
        counter+=1

    if supervised:
        img=data[index,...,0]
        imgClean=data[index,...,1]
        manipulate=False
    else:
        img=data[index]
        imgClean=img
        manipulate=True
        
    imgOut, imgOutC, mask = randomCrop(img, size, numPix,
                                      imgClean=imgClean,
                                      augment=augment,
                                      manipulate = manipulate )
    
    return imgOut, imgOutC, mask, counter

def randomCrop(img, size, numPix, imgClean=None, augment=True, manipulate=True):
    '''
    Cuts out a random crop from an image.
    Manipulates pixels in the image (N2V style) and produces the corresponding mask of manipulated pixels.
    Patches are augmented by randomly deciding to mirror them and/or rotating them by multiples of 90 degrees.
    
    Parameters
    ----------
    img: numpy array
        your dataset, should be a 2D image
    size: int
        witdth and height of the patch
    numPix: int
        The number of pixels that is to be manipulated/masked N2V style.
    dataClean(optinal): numpy array 
        This dataset could hold your target data e.g. clean images.
        If it is not provided the function will use the image from 'data' N2V style
    augment: bool
        should the patches be randomy flipped and rotated?
        
    Returns
    ----------    
    imgOut: numpy array 
        Cropped patch from training data with pixels manipulated N2V style.
    imgOutC: numpy array
        Cropped target patch. Pixels have not been manipulated.
    mask: numpy array
        An image marking which pixels have been manipulated (value 1) and which not (value 0).
        In N2V or PN2V only these pixels should be used to calculate gradients.
    '''
    
    assert img.shape[0] >= size
    assert img.shape[1] >= size

    x = np.random.randint(0, img.shape[1] - size)
    y = np.random.randint(0, img.shape[0] - size)

    imgOut = img[y:y+size, x:x+size].copy()
    imgOutC= imgClean[y:y+size, x:x+size].copy()
    
    maxA=imgOut.shape[1]-1
    maxB=imgOut.shape[0]-1
    
    if manipulate:
        mask=np.zeros(imgOut.shape)
        hotPixels=getStratifiedCoords2D(numPix,imgOut.shape)
        for p in hotPixels:
            a,b=p[1],p[0]

            roiMinA=max(a-2,0)
            roiMaxA=min(a+3,maxA)
            roiMinB=max(b-2,0)
            roiMaxB=min(b+3,maxB)
            roi=imgOut[roiMinB:roiMaxB,roiMinA:roiMaxA]
            a_ = 2
            b_ = 2
            while a_==2 and b_==2:
                a_ = np.random.randint(0, roi.shape[1] )
                b_ = np.random.randint(0, roi.shape[0] )

            repl=roi[b_,a_]
            imgOut[b,a]=repl
            mask[b,a]=1.0
    else:
        mask=np.ones(imgOut.shape)

    if augment:
        rot=np.random.randint(0,4)
        imgOut=np.array(np.rot90(imgOut,rot))
        imgOutC=np.array(np.rot90(imgOutC,rot))
        mask=np.array(np.rot90(mask,rot))
        if np.random.choice((True,False)):
            imgOut=np.array(np.flip(imgOut))
            imgOutC=np.array(np.flip(imgOutC))
            mask=np.array(np.flip(mask))


    return imgOut, imgOutC, mask

def trainingPred(my_train_data, net, dataCounter, size, bs, numPix, device, augment=True, supervised=True):
    '''
    This function will assemble a minibatch and process it using the a network.
    
    Parameters
    ----------
    my_train_data: numpy array
        Your training dataset, should be a stack of 2D images, i.e. a 3D numpy array
    net: a pytorch model
        the network we want to use
    dataCounter: int
        The index of the next image to be used. 
    size: int
        Witdth and height of the training patches that are to be used.
    bs: int 
        The batch size.
    numPix: int
        The number of pixels that is to be manipulated/masked N2V style.
    augment: bool
        should the patches be randomy flipped and rotated?
    Returns
    ----------
    samples: pytorch tensor
        The output of the network
    labels: pytorch tensor
        This is the tensor that was is used a target.
        It holds the raw unmanipulated patches.
    masks: pytorch tensor
        A tensor marking which pixels have been manipulated (value 1) and which not (value 0).
        In N2V or PN2V only these pixels should be used to calculate gradients.
    dataCounter: int
        The updated counter parameter, it is increased by one.
        When the counter reaches the end of the dataset, it is reset to zero and the dataset is shuffled.
    '''
    
    # Init Variables
    inputs= torch.zeros(bs,1,size,size)
    labels= torch.zeros(bs,size,size)
    masks= torch.zeros(bs,size,size)
   

    # Assemble mini batch
    for j in range(bs):
        im,l,m, dataCounter=randomCropFRI(my_train_data,
                                          size,
                                          numPix,
                                          counter=dataCounter,
                                          augment=augment,
                                          supervised=supervised)
        inputs[j,:,:,:]=utils.imgToTensor(im)
        labels[j,:,:]=utils.imgToTensor(l)
        masks[j,:,:]=utils.imgToTensor(m)

    # Move to GPU
    inputs_raw, labels, masks= inputs.to(device), labels.to(device), masks.to(device)

    # Move normalization parameter to GPU
    stdTorch=torch.Tensor(np.array(net.std)).to(device)
    meanTorch=torch.Tensor(np.array(net.mean)).to(device)
    
    # Forward step
    outputs = net((inputs_raw-meanTorch)/stdTorch) * 10.0 #We found that this factor can speed up training
    samples=(outputs).permute(1, 0, 2, 3)
    
    # Denormalize
    samples = samples * stdTorch + meanTorch
    
    return samples, labels, masks, dataCounter

def lossFunction(samples, labels, masks, std, psf, regularization, positivity_constraint):
    psf_shape = psf.shape[2]
    pad_size = (psf_shape - 1)//2
    assert psf.shape[2] == psf.shape[3]
    rs=torch.mean(samples,dim=0).reshape(samples.shape[1],1,samples.shape[2],samples.shape[3])
    # we pad the result
    rs=torch.nn.functional.pad(rs,(pad_size, pad_size, pad_size, pad_size),mode='reflect')
    # and convolve it with the psf
    conv=torch.nn.functional.conv2d(rs,
                                    weight=psf,
                                    padding=0,
                                    stride=[1,1])
    conv=conv.reshape(samples.shape[1],samples.shape[2],samples.shape[3])
    
    # This is the implementation of the positivity constraint
    signs = (samples < 0).float()
    samples_positivity_constraint = samples * signs
    
    # the N2V loss
    errors=(labels-conv)**2
    loss= torch.sum( errors *masks  ) /torch.sum(masks)
    
    # TV regularization
    yprior=((samples[:,:,2:,1:-1]-samples[:,:,:-2,1:-1])/2.0)**2
    xprior=((samples[:,:,1:-1,2:]-samples[:,:,1:-1,:-2])/2.0)**2
    reg=torch.sqrt(yprior+xprior+1e-15) # total variation
    
    return loss/(std**2) + (torch.mean(reg)*regularization + positivity_constraint * torch.mean(torch.abs(samples_positivity_constraint)))/std


def trainNetwork(net, trainData, valData, postfix, device,
                 directory='.',
                 numOfEpochs=200, stepsPerEpoch=50,
                 batchSize=4, patchSize=100, learningRate=0.0001,
                 numMaskedPixels=100*100/32.0, 
                 virtualBatchSize=20, valSize=20,
                 augment=True,
                 supervised=False,
                 psf=None, regularization = 0.0, positivity_constraint = 1.0):
    '''
    Train a network using 
    
    
    Parameters
    ----------
    net: 
        The network we want to train.
        The number of output channels determines the number of samples that are predicted.
    trainData: numpy array
        Our training data. A 3D array that is interpreted as a stack of 2D images.
    valData: numpy array
        Our validation data. A 3D array that is interpreted as a stack of 2D images.
    postfix: string
        This identifier is attached to the names of the files that will be saved during training.
    device: 
        The device we are using, e.g. a GPU or CPU
    directory: string
        The directory all files will be saved to.
    numOfEpochs: int
        Number of training epochs.
    stepsPerEpoch: int
        Number of gradient steps per epoch.
    batchSize: int
        The batch size, i.e. the number of patches processed simultainasly on the GPU.
    patchSize: int
        The width and height of the square training patches.
    learningRate: float
        The learning rate.
    numMaskedPixels: int
        The number of pixels that is to be manipulated/masked N2V style in every training patch.
    virtualBatchSize: int
        The number of batches that are processed before a gradient step is performed.
    valSize: int
        The number of validation patches processed after each epoch.
    augment: bool
        should the patches be randomy flipped and rotated? 
    supervised: bool
        Use this if you want to do supervised training instead of N2V
    psf: torch tensor
        This is the PSF that will be convolved with the predicted image during training
    regularization: float
        The weight for optional TV regularization of the deconvolved image.
    positivity_constraint: float
        The weight for the positivity constraint
        
    Returns
    ----------    
    trainHist: numpy array 
        A numpy array containing the avg. training loss of each epoch.
    valHist: numpy array
        A numpy array containing the avg. validation loss after each epoch.
    '''
        
    # Calculate mean and std of data.
    # Everything that is processed by the net will be normalized and denormalized using these numbers.
    combined=np.concatenate((trainData,valData))
    net.mean=np.mean(combined)
    net.std=np.std(combined)
    
    net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=learningRate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    running_loss = 0.0
    stepCounter=0
    dataCounter=0

    trainHist=[]
    valHist=[]
    
    while stepCounter / stepsPerEpoch < numOfEpochs:  # loop over the dataset multiple times
        losses=[]
        optimizer.zero_grad()
        stepCounter+=1

        # Loop over our virtual batch
        for a in range (virtualBatchSize):
            outputs, labels, masks, dataCounter = trainingPred(trainData,
                                                               net,
                                                               dataCounter,
                                                               patchSize, 
                                                               batchSize,
                                                               numMaskedPixels,
                                                               device,
                                                               augment = augment,
                                                               supervised = supervised)
            loss=lossFunction(outputs, labels, masks, net.std, psf, regularization, positivity_constraint)
            loss.backward()
            running_loss += loss.item()
            losses.append(loss.item())

        optimizer.step()

        # We have reached the end of an epoch
        if stepCounter % stepsPerEpoch == stepsPerEpoch-1:
            running_loss=(np.mean(losses))
            losses=np.array(losses)
            utils.printNow("Epoch "+str(int(stepCounter / stepsPerEpoch))+" finished")
            utils.printNow("avg. loss: "+str(np.mean(losses))+"+-(2SEM)"+str(2.0*np.std(losses)/np.sqrt(losses.size)))
            trainHist.append(np.mean(losses))
            losses=[]
            torch.save(net,os.path.join(directory,"last_"+postfix+".net"))

            valCounter=0
            net.train(False)
            losses=[]
            for i in range(valSize):
                outputs, labels, masks, valCounter = trainingPred(valData,
                                                                  net,
                                                                  valCounter,
                                                                  patchSize, 
                                                                  batchSize,
                                                                  numMaskedPixels,
                                                                  device,
                                                                  augment = augment,
                                                                  supervised = supervised)
                loss=lossFunction(outputs, labels, masks, net.std, psf, regularization, positivity_constraint)
                losses.append(loss.item())
            net.train(True)
            avgValLoss=np.mean(losses)
            if len(valHist)==0 or avgValLoss < np.min(np.array(valHist)):
                torch.save(net,os.path.join(directory,"best_"+postfix+".net"))
            valHist.append(avgValLoss)
            scheduler.step(avgValLoss)
            epoch= (stepCounter / stepsPerEpoch)
            np.save(os.path.join(directory,"history"+postfix+".npy"), (np.array( [np.arange(epoch),trainHist,valHist ] ) ) )

    utils.printNow('Finished Training')
    return trainHist, valHist
