""" Contains the Population class, which handles members, evolution, and breeding """

import tensorflow as tf
import random as r
from . import network as net

class Population:

    def __init__(self, pop_size, input_shape, random_init=True):
        self.MAX_LAYER_SIZE = 10
        self.MAX_LAYER_COUNT = 5
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
            for member in range(pop_size):
                # print("Member %d:", member+1)
                n = r.randint(3, self.MAX_LAYER_COUNT)
                layer_dim = [input_shape] + [r.randint(1, self.MAX_LAYER_SIZE) for i in range(n-1)]
                layer_activations = [None] + [self.map_activation[self.valid_activations[r.randint(0, len(self.valid_activations)-1)]] for i in range(n-1)]
                # print("\t%d layers", n)
                # print("\tLayer Dimensions: %s" % layer_dim)
                # print("\tLayer Activations: %s" % layer_activations)
                self.members.append(net.Network(
                    layer_dim,
                    layer_activations
                ))

    def __str__(self):
        formatted = "\n-----------------------------\n"
        for i, member in enumerate(self.members):
            formatted += ("Member %d:\n" % (i+1)) + str(member) + "\n-----------------------------\n"
        return formatted