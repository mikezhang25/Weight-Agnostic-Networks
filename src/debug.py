
import network as net
import population as pop
import gamemaster as gm
from parameters import *

mp = gm.GameMaster(2)
population = pop.Population(10, random_init=True, evaluator=mp)
population.evolve(10, save_progress=True, save_dir="./debug")

# print(len(population.get_mating_pool(population.get_normalized_fitness())))
"""
import gym
env = gym.make("FetchPickAndPlace-v1")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
"""
