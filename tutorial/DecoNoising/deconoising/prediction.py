############################################
#   Prediction
############################################

import numpy as np
import torch

from pn2v.utils import imgToTensor
from pn2v.utils import denormalize
from pn2v.utils import normalize
from scipy.ndimage import gaussian_filter


def artificial_psf(size_of_psf=81, std_gauss=1.0):
    filt = np.zeros((size_of_psf, size_of_psf))
    p = (size_of_psf - 1) // 2
    filt[p, p] = 1
    filt = torch.tensor(
        gaussian_filter(filt, std_gauss)
        .reshape(1, 1, size_of_psf, size_of_psf)
        .astype(np.float32)
    )
    filt = filt / torch.sum(filt)
    return filt


psf_tensor = artificial_psf(std_gauss=1.0)


def predict(net, im, device, outScaling):
    """
    Process an image using our network.

    Parameters
    ----------
    im: numpy array
        2D image we want to process
    net: a pytorch model
        the network we want to use
    device:
        The device your network lives on, e.g. your GPU
    outScaling: float
        We found that scaling the output by a factor (default=10) helps to speedup training.
    Returns
    ----------
    decoPred: numpy array
        Image containing the prediction.
    """
    # stdTorch = torch.Tensor(np.array(net.std)).to(device)
    # meanTorch = torch.Tensor(np.array(net.mean)).to(device)
    stdTorch = torch.Tensor(np.array(473.19403)).to(device)
    meanTorch = torch.Tensor(np.array(305.90112)).to(device)

    # im=(im-net.mean)/net.std
    inputs_raw = torch.zeros(1, 1, im.shape[0], im.shape[1])
    inputs_raw[0, :, :, :] = imgToTensor(im)

    # copy to GPU
    inputs_raw = inputs_raw.to(device)

    # normalize
    inputs = (inputs_raw - meanTorch) / stdTorch
    output = net(inputs)

    samples = (output).permute(
        1, 0, 2, 3
    ) * outScaling  # We found that this factor can speed up training
    samples = samples * stdTorch + meanTorch

    # denormalize
    decoPred = torch.mean(samples, dim=0, keepdim=True)[
        0, ...
    ]  # Sum up over all samples
    decoPred = decoPred.cpu().detach().numpy()
    decoPred.shape = (output.shape[2], output.shape[3])

    return decoPred, None


def tiledPredict(net, im, ps, overlap, device, pad=False, outScaling=10.0):
    """
    Tile the image to save GPU memory.
    Process it using our network.

    Parameters
    ----------
    im: numpy array
        2D image we want to process
    net: a pytorch model
        the network we want to use
    ps: int
        the widht/height of the square tiles we want to use in pixels
    overlap: int
        number of pixels we want the tiles to overlab in x and y
    device:
        The device your network lives on, e.g. your GPU
    outScaling: float
        We found that scaling the output by a factor (default=10) helps to speedup training.
    pad: bool
        (default=False)
        If True : We use padding for the last patch in each row/column.
        If False: (default) The last patch in each row/column is aligned to the right/bottom border of the image.
        This might have the potential to cause discontinuities.

    Returns
    ----------
    deconvolvedResult: numpy array
        Image containing the network prediction before the convolution
    denoisedResult: numpy array
        This is the result of convolving 'deconvolvedResult' woth the PSF.
    """
    # computing the deconvolved result
    if pad:
        deconvolvedResult = tiledPredict_pad(im, net, ps, overlap, device, outScaling)
    else:
        deconvolvedResult = tiledPredict_reflect(
            im, net, ps, overlap, device, outScaling
        )

    psf = psf_tensor.cpu()
    psf_shape = psf.shape[2]
    pad_size = (psf_shape - 1) // 2

    # We pad the result and then convolve it with the PSF to compute the denoised result
    deconvolved_ = torch.from_numpy(deconvolvedResult.astype(np.float32))
    deconvolved_ = deconvolved_.reshape(
        1, 1, deconvolvedResult.shape[0], deconvolvedResult.shape[1]
    )
    deconvolved_ = torch.nn.functional.pad(
        deconvolved_, (pad_size, pad_size, pad_size, pad_size), mode="reflect"
    )
    denoisedResult = torch.nn.functional.conv2d(
        deconvolved_, weight=psf, padding=0, stride=[1, 1]
    )[0, 0, :, :].numpy()

    return deconvolvedResult, denoisedResult


def tiledPredict_pad(im, net, ps, overlap, device, outScaling=10.0):
    """
    Tile the image to save GPU memory.
    Process it using our network.
    We use padding for the last patch in each row/column.

    Parameters
    ----------
    im: numpy array
        2D image we want to process
    net: a pytorch model
        the network we want to use
    ps: int
        the widht/height of the square tiles we want to use in pixels
    overlap: int
        number of pixels we want the tiles to overlab in x and y
    device:
        The device your network lives on, e.g. your GPU
    outScaling: float
        We found that scaling the output by a factor (default=10) helps to speedup training.

    Returns
    ----------
    decoPred: numpy array
        Image containing the prediction.
    """

    decoPred = np.zeros(im.shape)
    xmin = 0
    ymin = 0
    xmax = ps
    ymax = ps
    ovLeft = 0
    while xmin < im.shape[1]:
        ovTop = 0
        while ymin < im.shape[0]:
            inputPatch = im[ymin:ymax, xmin:xmax]
            padX = ps - inputPatch.shape[1]
            padY = ps - inputPatch.shape[0]

            inputPatch = np.pad(
                inputPatch,
                ((0, padY), (0, padX)),
                "constant",
                constant_values=(305.90112, 305.90112),
            )
            # inputPatch=np.pad(inputPatch,((0, padY),(0,padX)), 'reflect')
            a, b = predict(net, inputPatch, device, outScaling=outScaling)
            a = a[: a.shape[0] - padY, : a.shape[1] - padX]

            decoPred[ymin:ymax, xmin:xmax][ovTop:, ovLeft:] = a[ovTop:, ovLeft:]

            ymin = ymin - overlap + ps
            ymax = ymin + ps
            ovTop = overlap // 2
        ymin = 0
        ymax = ps
        xmin = xmin - overlap + ps
        xmax = xmin + ps
        ovLeft = overlap // 2

    return decoPred


def tiledPredict_reflect(im, net, ps, overlap, device, outScaling=10.0):
    """
    Tile the image to save GPU memory.
    Process it using our network.
    The last patch in each row/column is aligned to the right/bottom border of the image.
    This might have the potential to cause discontinuities.

    Parameters
    ----------
    im: numpy array
        2D image we want to process
    net: a pytorch model
        the network we want to use
    ps: int
        the widht/height of the square tiles we want to use in pixels
    overlap: int
        number of pixels we want the tiles to overlab in x and y
    device:
        The device your network lives on, e.g. your GPU
    outScaling: float
        We found that scaling the output by a factor (default=10) helps to speedup training.

    Returns
    ----------
    decoPred: numpy array
        Image containing the prediction.
    """

    decoPred = np.zeros(im.shape)
    xmin = 0
    ymin = 0
    xmax = ps
    ymax = ps
    ovLeft = 0
    while xmin < im.shape[1]:
        ovTop = 0
        while ymin < im.shape[0]:
            ymin_ = min(im.shape[0], ymax) - ps
            xmin_ = min(im.shape[1], xmax) - ps
            lastPatchShiftY = ymin - ymin_
            lastPatchShiftX = xmin - xmin_
            print(ymin_, ymax, xmin_, xmax)
            a, b = predict(
                net, im[ymin_:ymax, xmin_:xmax], device, outScaling=outScaling
            )
            decoPred[ymin:ymax, xmin:xmax][ovTop:, ovLeft:] = a[
                lastPatchShiftY:, lastPatchShiftX:
            ][ovTop:, ovLeft:]
            ymin = ymin - overlap + ps
            ymax = ymin + ps
            ovTop = overlap // 2
        ymin = 0
        ymax = ps
        xmin = xmin - overlap + ps
        xmax = xmin + ps
        ovLeft = overlap // 2

    return decoPred
