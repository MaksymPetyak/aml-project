import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, LeakyReLU, MaxPool2D
from .model import Model


WEIGHTS_PATH = 'models/jeffrey_df/2015_07_17_123003_PARAMSDUMP.pkl'


class JFnet(Model):

    def __init__(self, width=512, height=512):
        pass

    @staticmethod
    def build_model(width=512, height=512, filename=None,
                    n_classes=5, batch_size=None, p_conv=0.0):
        pass

    @staticmethod
    def get_img_dim(width, height):
        """Second input to JFnet consumes image dimensions
        division by 700 according to https://github.com/JeffreyDF/
        kaggle_diabetic_retinopathy/blob/
        43e7f51d5f3b2e240516678894409332bb3767a8/generators.py::lines 41-42
        """
        return np.vstack((width, height)).T / 700.
