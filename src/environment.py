""" Contains the test environment for evaluation """
import gym

class Environment:
    def __init__(self):
        self.env = gym.make('MountainCarContinuous-v0')
        self.env.render()

    def step(self, action):
        self.env.render()
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()
