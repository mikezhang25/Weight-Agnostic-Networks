# """ The main driver function for population testing and simulation  "

import population as pop
import gamemaster as gm

if __name__ == '__main__':
    mp = gm.GameMaster('MountainCarContinuous-v0', thread_num=2)
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
