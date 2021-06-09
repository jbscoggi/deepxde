from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from . import activations
from . import initializers
from . import regularizers
from .map import Map
from .. import config
from ..backend import tf
from ..utils import timing


class TFNN(Map):
    """Transformer Feed-forward neural networks from Wang et al. 2020.
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_neurons,
        num_layers,
        activation,
        kernel_initializer
    ):
        super(TFNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)

    @property
    def inputs(self):
        return self.x

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.y_

    @timing
    def build(self):
        print("Building transformer feed-forward neural network...")
        self.x = tf.placeholder(config.real(tf), [None, self.input_size])

        y = self.x
        if self._input_transform is not None:
            y = self._input_transform(y)
        
        u = self.dense(y, self.num_neurons, activation=self.activation)
        v = self.dense(y, self.num_neurons, activation=self.activation)

        for i in range(self.num_layers - 1):
            y = self.dense(y, self.num_neurons)
            y = self.activation(1-y)*u + self.activation(y)*v
        
        self.y = self.dense(y, self.output_size)
        
        if self._output_transform is not None:
            self.y = self._output_transform(self.x, self.y)

        self.y_ = tf.placeholder(config.real(tf), [None, self.output_size])
        self.built = True

    def dense(self, inputs, units, activation=None, use_bias=True):
        return tf.layers.dense(
            inputs,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.regularizer,
        )
