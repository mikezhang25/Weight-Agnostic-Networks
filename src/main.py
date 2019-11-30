""" The main driver function for population testing and simulation """

from parameters import *
import population as pop
import gamemaster as gm
import os
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
        resume = input("Resume Training? (Y/N)\n>>> ") == 'Y'

    print("Run Session Specifics:\n \
     || Population Size: %d\n \
     || Concurrent Training Space: %d \n \
     || Save progress at %s\n \
     || %s" % (
        args['popsize'],
        args['threadcount'],
        args['save'] if args['save'] else 'a random new directory',
        "Resuming Training" if resume else "Initializing Training Process"
    ))

    mp = gm.GameMaster(thread_num=args['threadcount'])
    crowd = pop.Population(0 if resume else args['popsize'], evaluator=mp)
    if resume:
        latest_gen = len(os.listdir("./" + args['save']))
        crowd.load_from_file("./" + args['save'] + "/gen-" + str(latest_gen+1) + ".txt")
        crowd.gen_count = latest_gen+1

    # norm_fitness = crowd.get_normalized_fitness()
    # mating_pool = crowd.get_mating_pool(norm_fitness)
    # print(sum(norm_fitness))
    # print(len(mating_pool))
    crowd.evolve(args["gencount"], save_dir="./autogen_dir_%d" % r.randint(0, 100) if not args['save'] else "./" + args['save'])
