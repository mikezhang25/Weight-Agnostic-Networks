""" The main driver function for population testing and simulation """

import population as pop
import gamemaster as gm
import os, os.path
import random as r
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evolves the optimal network structure for a gym environmenmt')
    parser.add_argument("-t", "--threadcount", type=int, required=True, help="Max number of processes allowed in optimization multiprocessing")
    parser.add_argument("-p", "--popsize", type=int, required=True, help="Size of network population")
    parser.add_argument("-g", "--gencount", type=int, required=True, help="Number of generations to evolve")
    parser.add_argument("-s", "--save", type=str, help="Directory to save the evolved generation map")
    args = vars(parser.parse_args())

    resume = False
    try:
        if args['save']:
            os.makedirs("./" + args['save'])
    except FileExistsError:
        resume = input("Resume Training? (Y/N)\n>>> ")

    mp = gm.GameMaster('MountainCarContinuous-v0', thread_num=args['threadcount'])
    crowd = pop.Population(0 if resume else args['popsize'], 1, 1, evaluator=mp)
    if resume:
        latest_gen = len([name for name in os.listdir("./" + args['save']) if os.path.isfile(name)])
        crowd.load_from_file("./" + args['save'] + "/gen-" + str(latest_gen+1) + ".txt")

    # norm_fitness = crowd.get_normalized_fitness()
    # mating_pool = crowd.get_mating_pool(norm_fitness)
    # print(sum(norm_fitness))
    # print(len(mating_pool))
    crowd.evolve(args["gencount"], save_dir="./autogen_dir_%d" % r.randint(0, 100) if not args['save'] else "./" + args['save'])
