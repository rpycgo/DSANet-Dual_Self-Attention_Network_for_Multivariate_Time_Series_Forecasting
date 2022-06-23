from ...config.config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Dropout
from tensorflow.keras.models import Model


class ARBlock(Layer):
    def __init__(self, config=model_config):
        super(ARBlock, self).__init__()
        self.config = config
        self.dense = Dense(units=config.h, activation='linear')

    def call(self, x):        
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.dense(x)
        output = tf.transpose(x, perm=(0, 2, 1))

        return output


def build_ar(x, config=model_config):
    input = Input(shape=x.shape[1:])

    output = ARBlock(config)(input)
    
    return Model(input, output)
