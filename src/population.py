""" Contains the Population class, which handles members, evolution, and breeding """

import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
import random as r
import progressbar
from parameters import *
import network as net

class Population:
    """
    Contains a fixed-size group of networks to evolve
    """

    def __init__(self, pop_size, random_init=True, evaluator=None):
        # autosave init parameters
        self.gen_count = 0
        self.n= pop_size
        self.members = []
        self.evaluator = evaluator

        if random_init:
            print("Creating Population...")
            bar = progressbar.ProgressBar(maxval=pop_size,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for member in range(pop_size):
                n = r.randint(2, MAX_NETWORK_SIZE)
                layer_dim = [r.randint(1, MAX_LAYER_SIZE) for i in range(n)]
                layer_activations = [VALID_ACTIVATIONS[r.randint(0, len(VALID_ACTIVATIONS)-1)] for i in range(n)]
                self.members.append(net.Network(
                    layer_dim,
                    layer_activations
                ))
                bar.update(member)
            bar.finish()

    def load_from_file(self, path_to_file):
        """
        Load population from a data file
        :param path_to_file: relative path to data file
        :return: best fitness
        """
        self.members = []
        print("Loading data from %s..." % path_to_file)
        f = open(path_to_file, "r")
        members = f.readlines()
        bar = progressbar.ProgressBar(maxval=len(members),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        best_fitness = None
        for i, member in enumerate(members):
            if not i:
                best_fitness = float(member.split(' ')[2])
                continue
            layer_sizes, layer_types = member.split(',')
            layer_sizes = [int(x) for x in layer_sizes.split('-')]
            layer_types = [x.rstrip('\n') if x != 'None' else None for x in layer_types.split('-')]
            self.members.append(net.Network(
                layer_sizes,
                layer_types
            ))
            bar.update(i)
        bar.finish()
        f.close()
        return best_fitness

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def evolve(self, step_num, save_progress=True, save_dir=''):
        """
        Evolves the population a user-defined number of steps
        :param step_num:
        :param save_progress:
        :param save_dir:
        :return:
        """
        for gen in range(step_num+1):
            try:
                os.makedirs(save_dir)
            except FileExistsError:
                pass
            self.step_gen(save_progress=save_progress, save_dir=save_dir + ("/gen-%d.txt" % (gen + 1)))

    def step_gen(self, save_progress=False, save_dir=''):
        """
        Evolves the current generation 1 generation
        :param save_progress: whether to save the current gen before evolving (False by default)
        :param save_dir: where to save the current gen ('' by default)
        :return: None
        """
        if self.evaluator == None:
            print("Evolution Error: No evaluator specified. Use <population>.set_evaluator() to specify evaluator")
            return

        # use roulette-wheel style of fitness scaling
        wheel = self.get_normalized_fitness()

        print("---------------------------------------------------")
        print("Evolving from Generation ", self.gen_count, "...")

        if save_progress:
            print("Saving Current Generation...")
            bar = progressbar.ProgressBar(maxval=len(self.members),
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()

            self.members.sort(key=lambda x: x.fitness, reverse=True)
            f = open(save_dir, "w+")
            f.write("Best Fitness: %.5f\n" % (self.members[0].fitness))
            for i, network in enumerate(self.members):
                f.write(str(network.get_printable()) + '\n')
                bar.update(i+1)
            bar.finish()
            f.close()
            print("Current Generation Successfully Saved")
            print("Best Fitness ", self.members[0].fitness)

        if ELITISM:
            new_pop = [self.members[0]]
            if self.n%2==0: new_pop.append(self.members[1])
        while len(new_pop) < self.n:
            # choose two parents at random
            a = self.pick_parent(wheel)
            while a >= len(self.members): a = self.pick_parent(wheel)
            b = self.pick_parent(wheel)
            while b == a or b >= len(self.members): b = self.pick_parent(wheel)
            parent_a = self.members[a]
            parent_b = self.members[b]

            child_a, child_b = self.cross(parent_a, parent_b)
            new_pop.append(child_a)
            new_pop.append(child_b)

        self.gen_count += 1
        self.members = new_pop
        print("Evolved Generation ", self.gen_count)
        print("---------------------------------------------------")

    def get_normalized_fitness(self):
        """
        Evaluates fitnesses of every member and ensures that they all sum to 100 (makes creating the pool easier)
        :return:
        """
        # evaluate fitness for every member
        fitnesses = self.evaluator.eval_fitnesses(self.members)
        for i, network in enumerate(self.members):
            network.fitness = fitnesses[i]

        # check if there were a mix of positive/negative
        displacement = min(fitnesses)
        if displacement <= 0:
            fitnesses = [val+abs(displacement)+1 for val in fitnesses]

        # create mating pool
        # normalize values to probabilities
        norm = [i/sum(fitnesses) for i in fitnesses]

        return norm

    def pick_parent(self, wheel):
        """
        Picks a random member from the roulette wheel
        :param wheel: array of roulette fitness weights
        :return: index of parent chosen
        """
        # scale the values up so that we can hit the last index as well
        val = r.random()*(1+abs(min(wheel)))
        sum = 0
        i = 0
        while sum < val and i <= len(wheel):
            if i < len(wheel): sum += wheel[i]
            i += 1
        return i-1

    def cross(self, parent_a, parent_b):
        # child a has same layer num as parent a
        layer_dims = []
        layer_types = []
        for index in range(len(parent_a.layer_dims)):
            # print("Configing Layer ", index)
            parent = r.randint(0, 1)
            # print("Choosing parent %s" % ('A' if parent else 'B'))
            if parent:
                layer_dims.append(parent_a.layer_dims[index])
                layer_types.append(parent_a.layer_types[index])
            else:
                layer_dims.append(parent_b.layer_dims[index%len(parent_b.layer_dims)])
                layer_types.append(parent_b.layer_types[index%len(parent_b.layer_dims)])
        child_a = net.Network(layer_dims, layer_types)

        layer_dims = []
        layer_types = []
        for index in range(len(parent_b.layer_dims)):
            # print("Configing Layer ", index)
            parent = r.randint(0, 1)
            # print("Choosing parent %s" % ('A' if parent else 'B'))
            if parent:
                layer_dims.append(parent_b.layer_dims[index])
                layer_types.append(parent_b.layer_types[index])
            else:
                layer_dims.append(parent_a.layer_dims[index % len(parent_a.layer_dims)])
                layer_types.append(parent_a.layer_types[index % len(parent_a.layer_dims)])
            # print("New Layer:\n\tSize: %d\n\tType: %s" % (layer_dims[len(layer_dims)-1], layer_types[len(layer_types)-1]))

        child_b = net.Network(layer_dims, layer_types)
        return child_a, child_b

    def __str__(self):
        formatted = "\n-----------------------------\n"
        for i, member in enumerate(self.members):
            formatted += ("Member %d:\n" % (i+1)) + str(member) + "\n-----------------------------\n"
        return formatted

if __name__ == '__main__':
    pop = Population(0, 1, 1)
    pop.load_from_file("./autogen_dir_9/gen-1.txt")
