# Reproducability Challenge
This repo contains code reproducing results of ["Leveraging uncertainty information from deep neural networks for disease detection"](https://www.nature.com/articles/s41598-017-17876-z])

Original code used in paper:

<https://github.com/chleibig/disease-detection>

Paper utilizes a pre-trained model described in this blogpost:

<http://jeffreydf.github.io/diabetic-retinopathy-detection/>

## Structure
For an example of how uncertainty is used see [example notebook](/example.ipynb)

The models are implemented using Keras and Python 3, for more details see [BCNN model](BCNN.py) and [JFNet model](JFNet.py)

The data is ommitted as it is too large to store on github, but the model weights are stored in "models" directory.

## Poster

![poster](poster.pdf)
