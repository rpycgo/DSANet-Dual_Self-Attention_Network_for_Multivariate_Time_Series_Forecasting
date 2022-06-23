from ...config.config import model_config

from tensorflow.keras.layers import Input, Layer, MultiHeadAttention, Dropout, Dense, LayerNormalization
from tensorflow.keras.models import Model


class AttentionBlock(Layer):
    def __init__(self, config=model_config):
        super(AttentionBlock, self).__init__()
        self.config = model_config
        self.attention = MultiHeadAttention(
            num_heads=config.n_heads, 
            key_dim=config.n_g//config.n_heads,
            value_dim=config.n_g//config.n_heads
            )
        self.layer_normalization = LayerNormalization()

        self.dropout = Dropout(rate=model_config.dropout_rate)

    def build(self, input_shape):
        self.dense = Dense(units=input_shape[-1], activation='relu')

    def call(self, x):
        attention_output = self.attention(x, x, x)
        x += attention_output
        x = self.layer_normalization(x)
        x = self.dropout(x)

        dense_output = self.dense(x)
        x += dense_output
        x = self.layer_normalization(x)
        x = self.dropout(x)

        return x


def build_attention(x, config=model_config):
    input = Input(shape=x.shape[1:])

    output = input
    for _ in range(config.attention_stacks):
        output = AttentionBlock(config)(output)
    
    return Model(input, output)
