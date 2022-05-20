# CycleGAN in Tensorflow 2

This repository contains my Tensorflow-2 replication of [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf), an unpaired image-to-image translation model. The code was written with reference to the [official implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) in PyTorch.

## Features

* ResNet-based generator (`resnset_generator`)
* PatchGAN-based discriminator (`n_layer_discriminator`)
* Shuffling of inputs to discriminators in the training phase (`ImagePool`)
* Various learning-rate schedulers
* Using various datasets from [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/cycle_gan)

## Current Limitations

* UNet-based generator not implemented
* PixelGAN-based discriminator not implemented
* Reduce-learning-rate-on-plateau scheduler not implemented
* Command-line interface not implemented
* Training speed 1~2x slower than the official implementation
