import numpy as np
import tensorflow as tf
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
    main_input = Input(
        shape=(height, width, 3),
        batch_size=batch_size,
    )

    # Note: for conv layers paper uses untie_biases=True
    # layer will have separate bias parameters for entire output
    # As a result, the bias is a 3D tensor.
    # Implemented as a Bias layer

    # no need to init weights as they will be loaded from a file
    # Conv layers(filters, kernel_size)
    conv_main_1 = Conv2D(
        32, 7, strides=(2, 2), padding='same',
        use_bias=False,
        activation=None,
    )(main_input)
    conv_bias_1 = Bias()(conv_main_1)
    conv_activation_1 = LeakyReLU(alpha=0.5)(conv_bias_1)
    dropout_1 = Dropout(p_conv)(conv_activation_1)
    maxpool_1 = MaxPool2D(pool_size=3, strides=(2, 2))(dropout_1)
    # 3
    conv_main_2 = Conv2D(
        32, 3, strides=(1, 1), padding='same',
        use_bias=False,
        activation=None,
    )(maxpool_1)
    conv_bias_2 = Bias()(conv_main_2)
    conv_activation_2 = LeakyReLU(alpha=0.5)(conv_bias_2)
    dropout_2 = Dropout(p_conv)(conv_activation_2)
    # 4
    conv_main_3 = Conv2D(
        32, 3, strides=(1, 1), padding='same',
        use_bias=False,
        activation=None,
    )(dropout_2)
    conv_bias_3 = Bias()(conv_main_3)
    conv_activation_3 = LeakyReLU(alpha=0.5)(conv_bias_3)
    dropout_3 = Dropout(p_conv)(conv_activation_3)
    maxpool_3 = MaxPool2D(pool_size=3, strides=(2, 2))(dropout_3)
    # 6
    conv_main_4 = Conv2D(
        64, 3, strides=(1, 1), padding='same',
        use_bias=False,
        activation=None,
    )(maxpool_3)
    conv_bias_4 = Bias()(conv_main_4)
    conv_activation_4 = LeakyReLU(alpha=0.5)(conv_bias_4)
    dropout_4 = Dropout(p_conv)(conv_activation_4)
    # 7
    conv_main_5 = Conv2D(
        64, 3, strides=(1, 1), padding='same',
        use_bias=False,
        activation=None,
    )(dropout_4)
    conv_bias_5 = Bias()(conv_main_5)
    conv_activation_5 = LeakyReLU(alpha=0.5)(conv_bias_5)
    dropout_5 = Dropout(p_conv)(conv_activation_5)
    maxpool_5 = MaxPool2D(pool_size=3, strides=(2, 2))(dropout_5)
    # 9
    conv_main_6 = Conv2D(
        128, 3, strides=(1, 1), padding='same',
        use_bias=False,
        activation=None,
    )(maxpool_5)
    conv_bias_6 = Bias()(conv_main_6)
    conv_activation_6 = LeakyReLU(alpha=0.5)(conv_bias_6)
    dropout_6 = Dropout(p_conv)(conv_activation_6)
    # 10
    conv_main_7 = Conv2D(
        128, 3, strides=(1, 1), padding='same',
        use_bias=False,
        activation=None,
    )(dropout_6)
    conv_bias_7 = Bias()(conv_main_7)
    conv_activation_7 = LeakyReLU(alpha=0.5)(conv_bias_7)
    dropout_7 = Dropout(p_conv)(conv_activation_7)
    # 11
    conv_main_8 = Conv2D(
        128, 3, strides=(1, 1), padding='same',
        use_bias=False,
        activation=None,
    )(dropout_7)
    conv_bias_8 = Bias()(conv_main_8)
    conv_activation_8 = LeakyReLU(alpha=0.5)(conv_bias_8)
    dropout_8 = Dropout(p_conv)(conv_activation_8)
    # 12
    conv_main_9 = Conv2D(
        128, 3, strides=(1, 1), padding='same',
        use_bias=False,
        activation=None,
    )(dropout_8)
    conv_bias_9 = Bias()(conv_main_9)
    conv_activation_9 = LeakyReLU(alpha=0.5)(conv_bias_9)
    dropout_9 = Dropout(p_conv)(conv_activation_9)
    maxpool_9 = MaxPool2D(pool_size=3, strides=(2, 2))(dropout_9)
    # layer 13

    # ------------------BCNN-Specialized-part-----------------------
    mean_pooled = GlobalAveragePooling2D(
        data_format='channels_last')(maxpool_9)
    max_pooled = GlobalMaxPooling2D(
        data_format='channels_last')(maxpool_9)
    global_pool = concatenate([mean_pooled, max_pooled], axis=1)

    softmax_input = Dense(
        units=n_classes, activation=None,)(global_pool)
    softmax_output = Softmax()(softmax_input)

    model = Model(inputs=[main_input], outputs=[softmax_output])

    return model

# testing with main
if __name__ == "__main__":
    model = BCNN_model()
    print(model.summary())
