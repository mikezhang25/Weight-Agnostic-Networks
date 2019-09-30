""" Contains the Population class, which handles members, evolution, and breeding """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
tf.logging.set_verbosity(tf.logging.ERROR)
deprecation._PRINT_DEPRECATION_WARNINGS = False
import random as r
import progressbar
import network as net

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Population:
    """
    Contains a fixed-size group of networks to evolve
    """

    def __init__(self, pop_size, input_shape, output_shape, random_init=True, evaluator=None):
        # random init params
        self.MAX_LAYER_SIZE = 100
        self.MAX_LAYER_COUNT = 20
        self.valid_activations = [
            "relu",
            "tanh",
            "sigmoid",
            "softmax"
        ]
        self.map_activation = {
            "relu": tf.nn.relu,
            "tanh": tf.nn.tanh,
            "sigmoid": tf.nn.sigmoid,
            "softmax": tf.nn.softmax
        }

        # autosave init parameters
        self.gen_count = 0
        self.inputs = input_shape
        self.outputs = output_shape
        self.members = []
        self.evaluator = evaluator

        if random_init:
            input_hidden_nodes = r.randint(1, self.MAX_LAYER_SIZE)
            print("Creating Population...")
            bar = progressbar.ProgressBar(maxval=pop_size,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for member in range(pop_size):
                n = r.randint(3, self.MAX_LAYER_COUNT)
                layer_dim = [input_hidden_nodes] + [r.randint(1, self.MAX_LAYER_SIZE) for i in range(n-1)]
                layer_activations = [None] + [self.map_activation[self.valid_activations[r.randint(0, len(self.valid_activations)-1)]] for i in range(n-1)]
                self.members.append(net.Network(
                    input_shape,
                    output_shape,
                    layer_dim,
                    layer_activations
                ))
                bar.update(member)
            bar.finish()

    def load_from_file(self, path_to_file):
        """
        Load population from a data file
        :param path_to_file: relative path to data file
        :return: None
        """
        self.members = []
        print("Loading data from %s..." % path_to_file)
        f = open(path_to_file, "r")
        members = f.readlines()
        bar = progressbar.ProgressBar(maxval=len(members),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i, member in enumerate(members):
            if not i:
                continue
            layer_sizes, layer_types = member.split(',')
            layer_sizes = [int(x) for x in layer_sizes.split('-')]
            layer_types = [self.map_activation[x.rstrip('\n')] if x != 'None' else None for x in layer_types.split('-')]
            self.members.append(net.Network(
                self.inputs,
                self.outputs,
                layer_sizes,
                layer_types
            ))
            bar.update(i)
        bar.finish()
        f.close()

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
            print("Population Error 1: No evaluator specified. Use <population>.set_evaluator() to specify evaluator")
            return

        print("---------------------------------------------------")
        print("Evolving from Generation ", self.gen_count, "...")
        norm_fitness = self.get_normalized_fitness()
        assert(sum(norm_fitness) != 0)

        if save_progress:
            print("Saving Current Generation...")
            bar = progressbar.ProgressBar(maxval=len(self.members),
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            self.members.sort(key=lambda x: x.fitness)
            f = open(save_dir, "w+")
            f.write("Best Fitness: %.5f\n" % (self.members[0].fitness))
            for i, network in enumerate(self.members):
                f.write(str(network.get_printable()) + '\n')
                bar.update(i+1)
            bar.finish()
            f.close()
            print("Current Generation Successfully Saved")
            print("Best Fitness ", self.members[0].fitness)
        # save members by order of fitness
        mating_pool = self.get_mating_pool(norm_fitness)

        new_pop = []
        while len(new_pop) < len(self.members):
            # choose two parents at random
            a = r.randint(0, len(mating_pool) - 1)
            b = r.randint(0, len(mating_pool) - 1)
            parent_a = mating_pool[a]
            parent_b = mating_pool[b]
            while parent_a == parent_b: parent_b = mating_pool[r.randint(0, len(mating_pool)-1)]

            child_a, child_b = self.cross(parent_a, parent_b)
            new_pop.append(child_a)
            new_pop.append(child_b)

        self.gen_count += 1
        print("Evolved Generation ", self.gen_count)
        print("---------------------------------------------------")

    def get_normalized_fitness(self):
        """

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
        # normalize values
        norm = [i/sum(fitnesses) for i in fitnesses]

        return norm

    def get_mating_pool(self, norm_fitness):
        # generate mating pools based on normalized probability
        mating_pool = []
        for i in range(len(norm_fitness)):
            for count in range(int(norm_fitness[i]*100)):
                mating_pool.append(self.members[i])

        return mating_pool

    def cross(self, parent_a, parent_b):
        # child a has same layer num as parent a
        layer_dims = []
        layer_types = []
        for index in range(len(parent_a.layer_dims)-1):
            # print("Configing Layer ", index)
            parent = r.randint(0, 1)
            # print("Choosing parent %s" % ('A' if parent else 'B'))
            if parent:
                layer_dims.append(parent_a.layer_dims[index])
                layer_types.append(parent_a.layer_types[index])
            else:
                layer_dims.append(parent_b.layer_dims[index%(len(parent_b.model.layers)-1)])
                layer_types.append(parent_b.layer_types[index%(len(parent_b.model.layers)-1)])
            # print("New Layer:\n\tSize: %d\n\tType: %s" % (layer_dims[len(layer_dims)-1], layer_types[len(layer_types)-1]))
        child_a = net.Network(parent_a.input_dim, parent_a.output_dim, layer_dims, layer_types)

        layer_dims = []
        layer_types = []
        for index in range(len(parent_b.layer_dims)-1):
            # print("Configing Layer ", index)
            parent = r.randint(0, 1)
            # print("Choosing parent %s" % ('A' if parent else 'B'))
            if parent:
                layer_dims.append(parent_b.layer_dims[index])
                layer_types.append(parent_b.layer_types[index])
            else:
                layer_dims.append(parent_a.layer_dims[index % (len(parent_a.model.layers) - 1)])
                layer_types.append(parent_a.layer_types[index % (len(parent_a.model.layers) - 1)])
            # print("New Layer:\n\tSize: %d\n\tType: %s" % (layer_dims[len(layer_dims)-1], layer_types[len(layer_types)-1]))

        child_b = net.Network(parent_b.input_dim, parent_b.output_dim, layer_dims, layer_types)
        return child_a, child_b

    def __str__(self):
        formatted = "\n-----------------------------\n"
        for i, member in enumerate(self.members):
            formatted += ("Member %d:\n" % (i+1)) + str(member) + "\n-----------------------------\n"
        return formatted

if __name__ == '__main__':
    pop = Population(0, 1, 1)
    pop.load_from_file("./autogen_dir_9/gen-1.txt")