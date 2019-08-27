# """ The main driver function for population testing and simulation  """
#
# import gym
# import multiprocessing as mpg
# import random as r
# import multiprocessing_gym as mpg
#
# def make_mp_envs(env_id, num_env, seed, start_idx=0):
#     def make_env(rank):
#         def fn():
#             env = gym.make(env_id)
#             env.seed(seed + rank)
#             return env
#         return fn
#     return mpg.SubprocVecEnv([make_env(i + start_idx) for i in range(num_env)])
#
# common_database = []
#
# class RGen:
#
#     def __init__(self, min, max):
#         self.min = min
#         self.max = max
#
#     def rand_number(self):
#         return 'Range is (%d, %d)' % (self.min, self.max)
#
# def spawn(index):
#     print(index)
#     # common_database.append(i)
#
# if __name__ == '__main__':
#     n = 5
#     generators = [RGen(i, i+1) for i in range(n)]
#     for i in range(n):
#         k = r.randint(0, 100)
#         p = mpg.Process(target=spawn, args=(k))
#         p.start()
#     print(common_database)
#     # item = make_mp_envs('MountainCarContinuous-v0', 5, 0)
#     # item.reset()
#     # print(item.step([0, 0, 0, 1, 1]))

# from multiprocessing import Pool
# from functools import partial
#
# class RGen:
#
#     def __init__(self, min, max):
#         self.min = min
#         self.max = max
#
#     def rand_number(self):
#         return 'Range is (%d, %d)' % (self.min, self.max)
#
# class MProcess:
#     def __init__(self, num_processes):
#         self.n = num_processes
#         self.gens = [RGen(i, i) for i in range(num_processes)]
#
#     def synth_network(self, input_dim, output_dim, layer_dims, layer_types):
#         import keras
#         return networks.Network(input_dim, output_dim, layer_dims, layer_types)
#
#     def job(self, specs):
#         import keras
#         import numpy as np
#         from keras.layers import Dense
#         import network
#         layer_dim = specs[0]
#         layer_types = specs[1]
#         model = network.Network(1, 1, layer_dim, layer_types)
#         # model = keras.Sequential()
#         # layer_dim = specs[0]
#         # layer_types = specs[1]
#         # model.add(Dense(layer_dim[0], activation=layer_types[0], input_shape=(1,), name="input_layer"))
#         #
#         # # add hidden and output layers
#         # for i in range(1, len(layer_dim)):
#         #     # self.model.add(Dropout(dropout, name="intermediate_%d"%i))
#         #     model.add(Dense(layer_dim[i], activation=layer_types[i], name="hidden_layer_%d" % i))
#         #
#         # # add activation for classification, no activation for float output
#         # model.add(Dense(1, name="output"))
#         return np.ndarray.tolist(model.run([1, 2]))
#
#     def get_gen_vals(self, networks):
#         pool = Pool(processes=self.n)
#         return pool.map(self.job, networks)
#
#     def close(self):
#         self.pool.close()

import population as pop
import gamemaster as gm

if __name__ == '__main__':
    mp = gm.GameMaster('MountainCarContinuous-v0', thread_num=4)
    crowd = pop.Population(8, 1, 1, evaluator=mp)
    crowd.evolve(5, save_dir='./test-1')

# import gamemaster, population, network
#
# if __name__ == '__main__':
#     gm = gamemaster.GameMaster('MountainCarContinuous-v0', thread_count=2)
#     # pop = population.Population(24, 1, 2)
#     networks = [network.Network(1, 1,
#                                 [64, 12, 22, 1],
#                                 ['relu', 'relu', 'softmax', 'softmax'],
#                                 ),
#                    network.Network(1, 1,
#                                    [64, 42, 12, 22, 1],
#                                    ['relu', 'relu', 'relu', 'softmax', 'softmax'],
#                                    ),
#                    network.Network(1, 1,
#                                    [64, 42, 12, 22, 1],
#                                    ['relu', 'relu', 'relu', 'softmax', 'softmax'],
#                                    ),
#                    network.Network(1, 1,
#                                    [64, 42, 12, 22, 1],
#                                    ['relu', 'relu', 'relu', 'softmax', 'softmax'],
#                                    )
#                    ]
#     gm.eval_fitnesses(networks, render=True)
#     # gm.eval_fitnesses(pop.members)
#     # multiprocessor = MProcess(12)
#     # print(multiprocessor.get_gen_vals("Fuck this", [i for i in range(11, 5, -1)]))
