from ...config.config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Conv2D, Dropout
from tensorflow.keras.models import Model


class GlobalTemporalConvolutionBlock(Layer):
    def __init__(self, config=model_config):
        super(GlobalTemporalConvolutionBlock, self).__init__()
        self.config = config
        self.conv = Conv2D(filters=config.n_g, kernel_size=config.time_seq, activation='relu')
        self.dropout = Dropout(rate=config.dropout_rate)

    def call(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        # to do: implements exact layers of merged by vectors
        x = tf.tile(x, [1, self.config.D, 1])

        return x


def build_global_temporal_convolution(x, config=model_config):
    input = Input(shape=x.shape[1:])

    output = GlobalTemporalConvolutionBlock(config)(input)
    
    return Model(input, output)
