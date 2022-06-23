from ...config.config import model_config
from ..layers.ar import build_ar
from ..layers.attention import build_attention
from ..layers.global_temporal_convolution import build_global_temporal_convolution
from ..layers.local_temporal_convolution import build_local_temporal_convolution

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def build_model(config=model_config):
    input = Input(shape=(config.time_seq, config.D))

    # global temporal convolution
    global_output = build_global_temporal_convolution(input, config)(input)
    global_output = build_attention(global_output, config)(global_output)
    global_output = Dense(units=config.D)(global_output)

    # local temporal convolution
    local_output = build_local_temporal_convolution(input, config)(input)
    local_output = build_attention(local_output, config)(local_output)
    local_output = Dense(units=config.D)(local_output)

    # AR
    ar_output = build_ar(input, config)(input)

    output = global_output + local_output + ar_output

    return Model(input, output)


class DASNet(Model):
    def __init__(self, config=model_config):
        super(DASNet, self).__init__()
        self.config = config
        self.model = build_model(config)

    def call(self, x):
        return self.model(x)
