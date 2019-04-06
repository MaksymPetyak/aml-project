import numpy as np
import tensorflow as tf

from Model import Model
from JFnet import JFnet

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dropout, Dense, MaxPool2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Input, LeakyReLU, Softmax, Reshape, Flatten
from tensorflow.keras.layers import concatenate, maximum, Lambda, Layer
import pickle


class Bias(Layer):
    """
    Adds bias to a layer. This is used for untied biases convolution.
    """
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(
            name='bias',
            shape=input_shape[1:],
            initializer='uniform',
            trainable=True)
        super(Bias, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.add(x, self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape


def feature_pool_max(input, pool_size=2, axis=1):
    """
    Based on lasagne implementation of FeaturePool
    """
    input_shape = input.shape.as_list()
    num_feature_maps = input_shape[axis]
    num_feature_maps_out = num_feature_maps // pool_size

    pool_shape = tf.TensorShape(
        (input_shape[1:axis] + [num_feature_maps_out, pool_size] + input_shape[axis+1:])
    )

    input_reshaped = Reshape(pool_shape)(input)
    # reduce along all axis but the target one
    reduction_axis = list(range(1, len(pool_shape)+1))
    reduction_axis.pop(axis-1)

    return tf.reduce_max(input_reshaped, axis=reduction_axis)

# create Layer for reshape with batchsize
# strong_reshape_func = lambda x: tf.reshape(x, (batch_size//2, concat.shape[1]*2))
# StrongReshape = Lambda(strong_reshape_func)
def BCNN_model(width=512, height=512, filename=None,
               n_classes=2, batch_size=64, p_conv=0.0,):
    """

    """
    jf_model = JFnet.build_model()
    conv_output = jf_model.get_layer("last_conv").output

    mean_pooled = GlobalAveragePooling2D(
        data_format='channels_last')(conv_output)
    max_pooled = GlobalMaxPooling2D(
        data_format='channels_last')(conv_output)
    global_pool = concatenate([mean_pooled, max_pooled], axis=1)

    softmax_input = Dense(
        units=n_classes, activation=None,)(global_pool)
    softmax_output = Softmax()(softmax_input)

    model = Model(inputs=[jf_model.input[0]], outputs=[softmax_output])

    return model

# testing with main
if __name__ == "__main__":
    model = BCNN_model()
    print(model.summary())
