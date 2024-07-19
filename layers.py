# Custom L1 Distance Layer Module

import tensorflow as tf
from tensorflow.keras import layers # type: ignore

class L1Dist(layers.Layer):
    def __init__(self, **kwargs):
        super(L1Dist, self).__init__(**kwargs)

    def call(self, inputs):
        # Ensure both inputs are tensors
        if not isinstance(inputs[0], tf.Tensor) or not isinstance(inputs[1], tf.Tensor):
            raise TypeError("Expected both inputs to be tensors.")

        # Calculate absolute difference
        return tf.math.abs(inputs[0] - inputs[1])