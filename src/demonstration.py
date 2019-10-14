""" Driver thread for visually showing the progress of an evolution session """

import os
import argparse
import gym
import population as pop
import gamemaster as gm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrates the best of a generation')
    parser.add_argument("-d", "--dir", type=str, required=True, help="Directory to load evolution files from")
    parser.add_argument("-t", "--ticks", type=int, help="Time steps to run the simulation")
    parser.add_argument("-g", "--gen", type=int, help="The non-latest generation you want to see")
    args = vars(parser.parse_args())

    crowd = pop.Population(0, 1, 1)

    # members are already in order of fitness
    latest_gen = len(os.listdir("./" + args['dir']))
    gen = args['gen'] if args['gen'] and args['gen'] <= latest_gen else latest_gen

    ticks = args['ticks'] if args['ticks'] else 1000

    best_fit = crowd.load_from_file("./" + args['dir'] + "/gen-" + str(gen) + ".txt")
    crowd.gen_count = latest_gen

    print("Now demonstrating the best of %s, generation %d, fitness %.5f" % (args['dir'], gen, best_fit))

    env = gym.make('MountainCarContinuous-v0')
    observations = env.reset()
    reward = -1

    for t in range(ticks):
        env.render()
        action = crowd.members[0].run(observations)
        observation, reward, done, info = env.step(action)
        if done:
            # print("Training Room %d finished up at time step %d" % (specs[2], t))
            break