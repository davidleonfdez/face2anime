# Unsupervised image to image translation with GANs

This project is a collection of experiments with generative adversarial networks (GANs).

It is focused on unsupervised image to image translation. All the experiments contained here for this task translate human to anime faces and viceversa, hence the name of the repository. However, you can use any pair of datasets.

To train a model, follow the steps detailed in [this notebook](notebooks/face2anime-bidirectional.ipynb).

To inspect some results, see [this notebook](notebooks/experiments/face2anime-bidirectional-17kds.ipynb).

There are also tools to perform image generation from random noise vectors (traditional GAN problem), even with relatively small datasets, using a simplification of the techniques presented in https://arxiv.org/abs/2006.06676. If you wish to take a look at an example that uses a small anime faces dataset, more centered in the effect of augmentations, see [this notebook](notebooks/experiments/noise2anime-gan.ipynb).


## Installation

In Linux:

`pip install requirements.txt`

In Windows:

`pip install win-requirements.txt`
