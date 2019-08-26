""" Contains the test environment for evaluation """
import gym

class GameMaster:
    def __init__(self):
        self.env = gym.make('MountainCarContinuous-v0')

    def eval_fitness(self, network, render=True):
        print("Evaluating Fitness...")
        observation = self.env.reset()
        reward = -1
        for t in range(1000):
            if render: self.env.render()
            observation, reward, done, info = self.env.step(network.get_category(network.run(observation)))
            # print(observation)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        self.env.close()
        print("Fitness is ", reward)
        return reward