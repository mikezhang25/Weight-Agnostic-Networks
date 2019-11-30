""" Contains Network class, which constitutes an individual in the population """

from parameters import *
import warnings
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Network:
    """
    Holds information for a network structure
    Actual network is constructed and destroyed at runtime to be compatible with multiprocessing
    """

    def __init__(self, layer_dim, layer_types):
        """
        Create blueprint for a network
        :param layer_dim: list of hidden layer dimensions
        :param layer_types: list of hidden layer activation func (min length 2)
        """
        if len(layer_dim) != len(layer_types):
            print("Network Initialization Error: Input layers and layer types mismatch")
            return

        # for evolution purposes, can't assign -1 because it could actually be a valid fitness
        self.fitness = None

        # store structure as sequential mapping of layer size to layer activation
        self.layer_dims = layer_dim
        self.layer_types = layer_types

    def build(self):
        """
        Constructs the keras model from network info
        :return: keras model
        """
        # put this here to make multiprocessing work
        # tells keras to stfu with its warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            import keras
            from keras.layers import Dense
            from keras import Sequential

        keras_model = Sequential()

        # input layer, must specify input shape
        keras_model.add(Dense(N_INPUTS, activation=None, input_shape=(INPUT_DIM,), name="input_layer",
                             kernel_initializer=keras.initializers.Constant(value=WEIGHT_CONSTANT)))

        # add hidden and output layers
        for i in range(len(self.layer_dims)):
            # self.model.add(Dropout(dropout, name="intermediate_%d"%i))
            keras_model.add(Dense(self.layer_dims[i], activation=self.layer_types[i], name="hidden_layer_%d" % i,
                                 kernel_initializer=keras.initializers.Constant(value=WEIGHT_CONSTANT)))

        # add activation for classification, no activation for float output
        keras_model.add(Dense(N_OUTPUTS, name="output",
                             kernel_initializer=keras.initializers.Constant(value=WEIGHT_CONSTANT)))

        return keras_model

    def strip_to_components(self, index):
        """
        Breaks down the network into it's init parameters
        :param index: index in population
        :return: [layer_dimensions, layer_activations, index_in_pop]
        """
        return [self.layer_dims, self.layer_types, index]

    def get_printable(self):
        """
        Returns savable and recoverable version of the network
        :return: None
        """
        formatted = ''
        for i, dim in enumerate(self.layer_dims):
            formatted += str(dim)
            formatted += "" if i == len(self.layer_dims) - 1 else "-"
        formatted += ","
        for i, layer in enumerate(self.layer_types):
            formatted += (layer if layer != None else "None") + ("" if i == len(self.layer_types)-1 else "-")
        return formatted

    def run(self, inputs):
        """
        Runs a set of data through the network
        :param inputs: input data (must fit input shape)
        :return: network outputs (analog)
        """
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
