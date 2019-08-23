""" Contains Network class, which constitutes an individual in the population """

import tensorflow as tf

class Network():
    def __init__(self, layer_dim, layer_types, print_graph=False):
        """
        Initialize Network based on manual layer dimensions and layer activations
        :param layer_dim: list of layer dimensions (min length 2)
        :param layer_types: list of tf layer activations (min length 2)
        """
        if len(layer_dim) < 2:
            print("Error N1: Insufficient Layer Size in Network Initialization")
            return
        elif len(layer_dim) != len(layer_types):
            print("Error N2: Input layers and layer types mismatch")
            return
        self.graph = tf.Graph()
        self.hidden_layers = []
        with self.graph.as_default():
            self.input_layer = tf.compat.v1.placeholder(name='input', shape=[None, layer_dim[0]], dtype = tf.float32)
            for i in range(1, len(layer_dim)):
                self.hidden_layers.append(tf.layers.dense(
                    self.input_layer if i == 1 else self.hidden_layers[i-2],
                    layer_dim[i],
                    activation=layer_types[i],
                    name=("hidden_layer_%d" % i) if i < len(layer_dim)-1 else "output"
                ))
            self.output_layer = self.hidden_layers.pop(len(self.hidden_layers)-1)



    def __str__(self):
        formatted = 'Input Layer: %s' % self.input_layer.shape
        for layer in self.hidden_layers:
            formatted += '\n%s: %s' % (layer.name, layer.shape)
        formatted += '\n%s: %s' % (self.output_layer.name, self.output_layer.shape)
        return formatted

