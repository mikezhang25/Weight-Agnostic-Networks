ENV_NAME = 'MountainCarContinuous-v0'

ELITISM = True

# max hidden layers allowed in a network
MAX_NETWORK_SIZE = 20
# max number of nodes allowed in a layer
MAX_LAYER_SIZE = 100

VALID_ACTIVATIONS = [
            'relu',
            'tanh',
            'sigmoid',
            'softmax'
        ]

# generally 1 and 1 unless mult-dimesional input e.g. pictures
INPUT_DIM = 1
OUTPUT_DIM = 1

# number of nodes in input layer
N_INPUTS = 4
# number of outputs expected (1 for analog outputs, specify number of categories for discreet)
N_OUTPUTS = 1
# keep network weights consistent
WEIGHT_CONSTANT = 0.5
