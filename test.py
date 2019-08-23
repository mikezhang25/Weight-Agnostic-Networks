import unittest as ut
import warnings

from src import network, population, environment

class EnvorionmentTest(ut.TestCase):
    def test_network_creation(self):
        """
        Verifies that Network constructors are working
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
                'softmax',
                'relu',
                'tanh',
                'sigmoid',
                'softmax'
            ],
        print_graph=True)

    def test_population_creation(self):
        """
        Verifies that Population constructors are working
        """
        pop = population.Population(5, 5)
        print(pop)

    def test_env_creation(self):
        """
        Verifies that Environment constructors are working
        """
        simulator = environment.Environment()
        for episode in range(10):
            observation = simulator.reset()
            for t in range(1000):
                print(observation)
                action = simulator.env.action_space.sample()
                observation, reward, done, info = simulator.step(action)
                if done:
                    print("Episode finished after {} timespaces".format(t+1))
                    break
        simulator.close()


if __name__ == "__main__":
    ut.main()