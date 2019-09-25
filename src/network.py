""" Contains Network class, which constitutes an individual in the population """

class Network:

    def __init__(self, input_dim, output_n, layer_dim, layer_types, dropout=0.5, print_graph=False):
        # put this here to make multiprocessing work
        import keras
        from keras.layers import Dense
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
        self.WEIGHT_CONSTANT = 0.5
        self.fitness = -1
        self.input_dim = input_dim
        self.output_dim = output_n
        self.layer_dims = layer_dim
        self.layer_types = layer_types

        # input layer, must specify input shape
        self.model.add(Dense(layer_dim[0], activation=layer_types[0], input_shape=(input_dim,), name="input_layer"))

        # add hidden and output layers
        for i in range(1, len(layer_dim)):
            # self.model.add(Dropout(dropout, name="intermediate_%d"%i))
            self.model.add(Dense(layer_dim[i], activation=layer_types[i], name="hidden_layer_%d"%i))

        # add activation for classification, no activation for float output
        self.model.add(Dense(output_n, name="output"))

        # keep weights the same

        print("Network Initialization Success")

        if print_graph: self.model.summary()

    def get_category(self, run_results):
        run_results = list(run_results)[0]
        return [max(run_results)]

    def strip_to_components(self, index):
        return [self.layer_dims, self.layer_types, index]

    def get_printable(self):
        formatted = ''
        for i, dim in enumerate(self.layer_dims):
            formatted += str(dim)
            formatted += "" if i == len(self.layer_dims) - 1 else "-"
        formatted += ","
        for i, layer in enumerate(self.layer_types):
            formatted += (layer.__name__ if layer != None else "None") + ("" if i == len(self.layer_types)-1 else "-")
        return formatted

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