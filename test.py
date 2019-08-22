import unittest as ut
import tensorflow as tf

from src import network

class NetworkTest(ut.TestCase):
    def test_network_creation(self):
        """
        Verifies that constructors are working
        """
        nn = network.Network(
            # Network dimensions
            [
                5,
                4,
                10,
                3,
                9
            ],
            [
                None,
                tf.nn.relu,
                tf.nn.tanh,
                tf.nn.sigmoid,
                tf.nn.softmax
            ])
        print(nn)