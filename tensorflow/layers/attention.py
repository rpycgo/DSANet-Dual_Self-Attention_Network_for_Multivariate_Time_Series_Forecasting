from ...config.config import model_config

from tensorflow.keras.layers import Input, Layer, Attention, Dropout, Dense
from tensorflow.keras.models import Model


class AttentionBlock(Layer):
    def __init__(self, ff_dense, config=model_config):
        super(AttentionBlock, self).__init__()
        self.config = model_config
        self.attention = Attention(use_scale=True)
        self.dense = Dense(units=ff_dense, activation='relu')
        self.dropout = Dropout(rate=model_config.dropout_rate)

    def call(self, x):
        attention_output = self.attention([x, x, x])
        x += attention_output
        x = self.dropout(x)

        dense_output = self.dense(x)
        x += dense_output
        x = self.dropout(x)

        return x


def build_attention(x, config=model_config):
    input = Input(shape=x.shape[1:])

    output = input
    for _ in range(config.attention_stacks):
        output = AttentionBlock(output.shape[-1], config)(output)
    
    return Model(input, output)
