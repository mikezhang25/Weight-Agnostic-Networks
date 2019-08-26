""" Contains Network class, which constitutes an individual in the population """

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy as np

class Network():

    def __init__(self, input_dim, output_n, layer_dim, layer_types, dropout=0.5, print_graph=False):
        """
        Initialize Network based on manual layer dimensions and layer activations
        :param (int)        input_dim: specifies n-dimension input vector
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
        self.fitness = -1
        self.input_dim = input_dim
        self.output_dim = output_n
        self.layer_dims = layer_dim
        self.layer_types = layer_types

        # input layer, must specify input shape
        self.model.add(Dense(layer_dim[0], activation=layer_types[0], input_shape=(input_dim,), name="input_layer"))

        # add hidden and output layers
        for i in range(1, len(layer_dim)):
            #self.model.add(Dropout(dropout, name="intermediate_%d"%i))
            self.model.add(Dense(layer_dim[i], activation=layer_types[i], name="hidden_layer_%d"%i))

        # add activation for classification, no activation for float output
        self.model.add(Dense(output_n, name="output"))

        print("Network Initialization Success")

        if print_graph: self.model.summary()

    def get_category(self, run_results):
        run_results = list(run_results)[0]
        return [max(run_results)]

    def run(self, inputs):
        return self.model.predict(inputs)

    def tune_weights(self, train_data, train_labels, epochs=1, batch_size=64):
        # specify training params: optimizer (for problem type), loss function, metrics (to train by)
        self.model.compile(optimizer='rmsprop',
                           loss='sparse_categorical_crossentropy', # default for multi-class classification
                           metrics=['accuracy'])

        # train the model
        self.model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

    def test(self, test_data, test_labels, batch_size=64):
        return self.model.evaluate(test_data, test_labels, batch_size=batch_size)

    def __str__(self):
        layer_format = "%s:\n\tLayer Size: %s\n\tLayer Type: %s\n" # % (layer_name, layer_shape, layer_type)
        formatted = ""
        for layer in self.model.layers:
            if str(type(layer)) == "<class 'keras.layers.core.Dropout'>": continue
            formatted += layer_format % (layer.name, layer.output_shape[1], str(type(layer)).split('\'')[1].split('.')[3])
        return formatted



# Legacy changes too chicken to delete
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