""" Contains the Population class, which handles members, evolution, and breeding """

import tensorflow as tf
from . import network as net

class Population:

    def __init__(self, pop_size, random_init=True):
        self.MAX_LAYER_SIZE = 10
        self.members = []
        self.valid_activations = [
            "relu",
            "tanh",
            "sigmoid",
            "softmax"
        ]
        self.map_activation = {
            "relu"   :tf.nn.relu,
            "tanh"   :tf.nn.tanh,
            "sigmoid":tf.nn.sigmoid,
            "softmax":tf.nn.softmax
        }
        if random_init:
            for i in range(pop_size):
                self.members.append(net.Network())