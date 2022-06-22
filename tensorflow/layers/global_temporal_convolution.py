from ...config.config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Conv1D, Dropout
from tensorflow.keras.models import Model


class GlobalTemporalConvolutionBlock(Layer):
    def __init__(self, config=model_config):
        super(GlobalTemporalConvolutionBlock, self).__init__()
        self.config = model_config
        self.conv = Conv1D(filters=config.n_g, kernel_size=config.time_seq, activation='relu')
        self.dropout = Dropout(rate=model_config.dropout_rate)

    def call(self, x):
        x = self.conv(x)
        x = tf.transpose(x, perm=(0, 2, 1))
       
        return x


def build_global_temporal_convolution(x, config=model_config):
    input = Input(shape=x.shape[1:])

    output = GlobalTemporalConvolutionBlock(config)(input)
    
    return Model(input, output)
