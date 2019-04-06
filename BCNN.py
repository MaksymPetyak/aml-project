import numpy as np
import tensorflow as tf

from Model import Model
from JFnet import JFnet

from tensorflow.keras.layers import Conv2D, Dropout, Dense, MaxPool2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Input, LeakyReLU, Softmax, Reshape, Flatten
from tensorflow.keras.layers import concatenate, maximum, Lambda, Layer
import pickle


class BCNN(Model):
    """Bayesian convolutional neural network (if p != 0 and on at test time)"""

    def __init__(self, p_conv=0.2, last_layer='13', weights=None,
                 n_classes=2, **kwargs):
        jf_model = JFnet.build_model(
            width=512, height=512,
            filename=JFnet.WEIGHTS_PATH,
            p_conv=p_conv, **kwargs)
        # remove unused layers
        conv_output = jf_model.get_layer("last_conv").output

        mean_pooled = GlobalAveragePooling2D(
            data_format='channels_last')(conv_output)
        max_pooled = GlobalMaxPooling2D(
            data_format='channels_last')(conv_output)
        global_pool = concatenate([mean_pooled, max_pooled], axis=1)

        softmax_input = Dense(
            units=n_classes, activation=None,)(global_pool)
        softmax_output = Softmax()(softmax_input)

        model = tf.keras.Model(
            inputs=[jf_model.input[0]],
            outputs=[softmax_output])

        # TODO: implement saving weights
        if weights is not None:
            pass

        super(BCNN, self).__init__(net=model)


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
    bcnn = BCNN(batch_size=2)
    bcnn.print_summary()
    model = bcnn.net
    input = np.zeros(model.input_shape)
    print("-" * 10 + "Predict" + "-" * 10)
    print(bcnn.predict(input))
    print("-" * 10 + "MC samples" + "-" * 10)
    print(bcnn.mc_samples(input))
