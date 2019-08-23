""" Contains Network class, which constitutes an individual in the population """

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

class Network():
    def __init__(self, input_shape, layer_dim, layer_types, dropout=0.5, print_graph=False):
        """
        Initialize Network based on manual layer dimensions and layer activations
        :param (int)        input_shape: specifies n-dimension input vector
        :param (int[])      layer_dim: list of layer dimensions (min length 2)
        :param (string[])   layer_types: list of layer activation func (min length 2)
        :param (float)      dropout: dropout probability
        :param (boolean)    print_graph: whether to print graph after init
        """
        if len(layer_dim) < 2:
            print("Network Initialization Error: Insufficient Layer Size in Network Initialization")
            return
        elif len(layer_dim) != len(layer_types):
            print("Network Initialization Error: Input layers and layer types mismatch")
            return
        self.model = keras.Sequential()

        # input layer, must specify input shape
        self.model.add(Dense(layer_dim[0], activation=layer_types[0], input_dim=input_shape))

        # add hidden and output layers
        for i in range(1, len(layer_dim)):
            self.model.add(Dropout(dropout))
            self.mode.add(Dense(layer_dim[i], activation=layer_types[i]))

        if print_graph: self.model.summary()


    # def __init__(self, layer_dim, layer_types, print_graph=False):
    #     """
    #     Initialize Network based on manual layer dimensions and layer activations
    #     :param layer_dim: list of layer dimensions (min length 2)
    #     :param layer_types: list of tf layer activations (min length 2)
    #     """
    #     if len(layer_dim) < 2:
    #         print("Error N1: Insufficient Layer Size in Network Initialization")
    #         return
    #     elif len(layer_dim) != len(layer_types):
    #         print("Error N2: Input layers and layer types mismatch")
    #         return
    #     self.graph = tf.Graph()
    #     self.hidden_layers = []
    #     with self.graph.as_default():
    #         self.input_layer = tf.compat.v1.placeholder(name='input', shape=[None, layer_dim[0]], dtype = tf.float32)
    #         for i in range(1, len(layer_dim)):
    #             self.hidden_layers.append(tf.layers.dense(
    #                 self.input_layer if i == 1 else self.hidden_layers[i-2],
    #                 layer_dim[i],
    #                 activation=layer_types[i],
    #                 name=("hidden_layer_%d" % i) if i < len(layer_dim)-1 else "output"
    #             ))
    #         self.output_layer = self.hidden_layers.pop(len(self.hidden_layers)-1)

    def run(self, inputs):
        pass

