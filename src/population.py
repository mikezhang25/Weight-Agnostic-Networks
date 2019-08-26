""" Contains the Population class, which handles members, evolution, and breeding """

import tensorflow as tf
import random as r
from . import network as net

class Population:

    def __init__(self, pop_size, input_shape, output_shape, random_init=True, evaluator=None):
        self.MAX_LAYER_SIZE = 100
        self.MAX_LAYER_COUNT = 20
        self.gen_count = 0
        self.inputs = input_shape
        self.outputs = output_shape
        self.members = []
        self.evaluator = evaluator
        self.valid_activations = [
            "relu",
            "tanh",
            "sigmoid",
            "softmax"
        ]
        self.map_activation = {
            "relu"   :tf.nn.relu,
            "tanh"   :tf.nn.tanh,
            "sigmoid":tf.nn.sigmoid,
            "softmax":tf.nn.softmax
        }
        if random_init:
            input_hidden_nodes = r.randint(1, self.MAX_LAYER_SIZE)
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

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def step_gen(self):
        if self.evaluator == None:
            print("Population Error 1: No evaluator specified. Use <population>.set_evaluator() to specify evaluator")
            return

        print("Evolving from Generation ", self.gen_count, "...")
        norm_fitness = self.get_normalized_fitness()

        mating_pool = self.get_mating_pool(norm_fitness)

        new_pop = []
        while len(new_pop) < len(self.members):
            # choose two parents at random
            parent_a = mating_pool[r.randint(0, len(mating_pool)-1)]
            parent_b = mating_pool[r.randint(0, len(mating_pool)-1)]
            while parent_a == parent_b: parent_b = mating_pool[r.randint(0, 99)]

            child_a, child_b = self.cross(parent_a, parent_b)
            new_pop.append(child_a)
            new_pop.append(child_b)

        self.gen_count += 1
        print("Evolved Generation ", self.gen_count)

    # TODO: Verify that positive-negative fitness mix doesn't screw up the norm
    def get_normalized_fitness(self):
        # evaluate fitness for every member
        fitnesses = []
        for network in self.members:
            network.fitness = self.evaluator.eval_fitness(network)
            fitnesses.append(network.fitness)

        # check if there were a mix of positive/negative
        displacement = min(fitnesses)
        if displacement <= 0:
            fitnesses = [val+abs(displacement)+1 for val in fitnesses]

        # create mating pool
        # normalize values
        norm = [float(i) / sum(fitnesses) for i in fitnesses]

        return norm

    def get_mating_pool(self, norm_fitness):
        # generate mating pools based on normalized probability
        mating_pool = []
        for i in range(len(self.members)-1):
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