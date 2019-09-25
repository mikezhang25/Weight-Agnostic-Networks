import unittest as ut
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

from src import network, population, gamemaster

class EnvorionmentTest(ut.TestCase):
    def test_network_creation(self):
        """
        Verifies that Network constructors are working
        """
        nn = network.Network(
            1, 1,
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
            ]
        )
        print("Network Creation is Online...")

    # def test_network_training(self):
    #     # test data from keras doc for Sequential model
    #     x_test = np.random.random((1, 2))
    #     y_test = keras.utils.to_categorical(np.random.randint(0, 1, size=(1, 1)), num_classes=2)
    #
    #     model = network.Network(1, 1,
    #                             [64, 42, 12, 22, 1],
    #                             ['relu', 'relu', 'relu', 'softmax', 'softmax'],
    #                             )
    #     # model.tune_weights(x_test[0], y_test[0], epochs=20, batch_size=128)
    #     result = model.run([4, 2])
    #     print("Testing Data: ", x_test)
    #     print("Testing Labels: ", y_test)
    #     print("Network output: ", result)
    #     print("Network category: ", model.get_category(result))
    #     # print(model.test(x_test, y_test, batch_size=128))

    def test_population_creation(self):
        """
        Verifies that Population constructors are working
        """
        pop = population.Population(5, 1, 2)
        print("Population Creation is Online...")

    def test_network_env_integration(self):
        """
        Verifies that Environment constructors are working
        """
        simulator = environment.Environment()
        observation = simulator.reset()

        pop = population.Population(10, 1, 2)

        for i, model in enumerate(pop.members):
            print("Member ", i)
            observation = simulator.env.reset()
            for t in range(1000):
                output = model.run(observation)
                action = model.get_category(output)
                random_action = simulator.env.action_space.sample()
                observation, reward, done, info = simulator.step(action)
                # print("Observation: ", observation)
                # print("Reward: ", reward)
                # print("Network Output: ", output)
                # print("Network Action: ", action)
                # print("Random Action: ", random_action)

                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        simulator.close()

    def test_load_pop_from_file(self):
        pop = population.Population(0, 1, 1)
        pop.load_from_file("./autogen_dir_9/gen-1.txt")

    def test_crossover(self):
        parent_a = network.Network(1, 2,
                                  [11, 9, 7, 5, 3],
                                  ['relu', 'relu', 'relu', 'softmax', 'softmax'],
                                  )
        parent_b = network.Network(1, 2,
                                   [10, 8, 6],
                                   ['relu', 'softmax', 'softmax'],
                                   )
        print("---------------------------------------")
        print("---------------------------------------")
        # for i in parent_a.model.get_weights():
        #     print(i)
        #     print("--------------")
        # print(parent_a.model.layers)
        pop = population.Population(0, 1, 2)
        child_a, child_b = pop.cross(parent_a, parent_b)
        print("Child A:\n", child_a)
        print("---------------------------------------")
        print("Child B:\n", child_b)

    def test_evolution(self):
        env = gamemaster.GameMaster('MountainCarContinuous-v0')
        pop = population.Population(20, 1, 1, evaluator=env)
        pop.step_gen()


if __name__ == "__main__":
    ut.main()