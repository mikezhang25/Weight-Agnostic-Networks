import unittest as ut
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

from src import network, population, environment

class EnvorionmentTest(ut.TestCase):
    # def test_network_creation(self):
    #     """
    #     Verifies that Network constructors are working
    #     """
    #     nn = network.Network(
    #         1,
    #         # Network dimensions
    #         [
    #             5,
    #             4,
    #             10,
    #             3,
    #             9
    #         ],
    #         [
    #             'softmax',
    #             'relu',
    #             'tanh',
    #             'sigmoid',
    #             'softmax'
    #         ],
    #     print_graph=True)

    def test_network_training(self):
        # # test data from keras doc for Sequential model
        x_train = np.random.random((1000, 20))
        y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
        x_test = np.random.random((100, 20))
        y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

        model = network.Network(20,
                                [64, 46, 8, 29, 10],
                                ['relu', 'relu', 'relu', 'relu', 'softmax'],
                                print_graph=True)

        model.tune_weights(x_train, y_train, epochs=20, batch_size=128)
        print(model.test(x_test, y_test, batch_size=128))

    # def test_population_creation(self):
    #     """
    #     Verifies that Population constructors are working
    #     """
    #     pop = population.Population(5, 5)
    #     print(pop)
    #
    # def test_env_creation(self):
    #     """
    #     Verifies that Environment constructors are working
    #     """
    #     simulator = environment.Environment()
    #     for _ in range(10):
    #         simulator.step(simulator.env.action_space.sample())
    #     simulator.close()


if __name__ == "__main__":
    ut.main()