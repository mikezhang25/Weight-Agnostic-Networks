""" Contains the test environment for evaluation """
import gym
from multiprocessing import Pool
from functools import partial
import progressbar
from parameters import *

class GameMaster:
    def __init__(self, thread_num=12):
        self.env = gym.make(ENV_NAME)

    def eval_fitnesses(self, networks, time_steps=1000):
        fitnesses = []
        print("Evaluating Population")
        bar = progressbar.ProgressBar(maxval=len(networks),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        # TODO: Make this bar global so can have closer analog output
        for i, network in enumerate(networks):
            observations = self.env.reset()
            model = network.build()
            reward = -1
            for t in range(time_steps):
                action = model.predict(observations)
                observation, reward, done, info = self.env.step(action)
                if done:
                    # print("Training Room %d finished up at time step %d" % (specs[2], t))
                    break
            fitnesses.append(reward)
            bar.update(i)
        bar.finish()
        return fitnesses

    def eval(self, time_steps, network_skeleton):
        network = network_skeleton.build()
        env = None
        env_id = -1
        while not env:
            for i in range(self.n):
                if self.env_free[i]:
                    env = self.envs[i]
                    env_id = i
                    self.env_free[i] = False
                    break

        # print("Starting Training Room %d" % specs[2])
        observations = env.reset()
        reward = -1
        for t in range(time_steps):
            action = network.predict(observations)
            observation, reward, done, info = env.step(action)
            if done:
                # print("Training Room %d finished up at time step %d" % (specs[2], t))
                break

        self.env_free[env_id] = True
        # destroy the network to save space
        del network
        return reward
