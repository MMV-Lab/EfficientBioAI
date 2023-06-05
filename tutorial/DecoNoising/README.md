# DecoNoising: Improving Blind Spot Denoising for Microscopy

We copied the original [repo](https://github.com/juglab/DecoNoising/tree/master/deconoising) here and add a new jupyter notebook called `Convallaria-Compression.ipynb` to show how to compress the model.

------------------
Anna S. Goncharova, Alf Honigmann, Florian Jug, and Alexander Krull</br>
Max Planck Institute of Molecular Cell Biology and Genetics (**[MPI-CBG](https://www.mpi-cbg.de/home/)**) <br>
Center for Systems Biology (**[CSBD](https://www.csbdresden.de/)**) in Dresden, Germany 

You can find the paper [here](https://arxiv.org/abs/2008.08414).

### Abstract 
Many microscopy applications are limited by the total amount of usable light and are consequently challenged by the resulting levels of noise in the acquired images.
This problem is often addressed via (supervised) deep learning based denoising. 
Recently, by making assumptions about the noise statistics, self-supervised methods have emerged.
Such methods are trained directly on the images that are to be denoised and do not require additional paired training data.
While achieving remarkable results, self-supervised methods can produce high-frequency artifacts and achieve inferior results compared to supervised approaches.
Here we present a novel way to improve the quality of self-supervised denoising.
Considering that light microscopy images are usually diffraction-limited, we propose to include this knowledge in the denoising process.
We assume the clean image to be the result of a convolution with a point spread function (PSF) and explicitly include this operation at the end of our neural network.
As a consequence, we are able to eliminate high-frequency artifacts and achieve self-supervised results that are very close to the ones achieved with traditional supervised methods.

### Requirements
We tested DecoNoising with `pytorch 1.0.1` using the `py3.7_cuda9.0.176_cudnn7.4.2_2` build.

### Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{goncharova2020,
  title={Improving Blind Spot Denoising for Microscopy},
  author={Goncharova, Anna S. and Honigmann, Alf and Jug, Florian and Krull, Alexander},
  booktitle={Proceedings of the European Conference on Computer Vision, Workshops},
  year={2020}
}
```

### Getting Started
You can try out our example notebooks:
`Convallaria-Training.ipynb` -- this notebook will automatically download example data and train a DecoNoising network training.
`Convallaria-Prediction.ipynb` -- this notebook will load the previously trained network and produces deconvolved and denoised results. The latter are compared to Ground Truth data.

