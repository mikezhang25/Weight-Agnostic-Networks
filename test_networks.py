import pytest

import src.network as net
import src.population as pop
import src.gamemaster as gm

def test_creation():
    """
    Verifies that Network constructors are working
    """
    nn = net.Network(
        1, 1,
        # Network dimensions
        [5,         4,      10,     3,          9] ,
        ['softmax', 'relu', 'tanh', 'sigmoid',  'softmax']
    )
    assert nn != None, "Network was not created successfully"
    print("Network Creation is Online...")

def test_crossover():
    """
    Verifies that the population crossover function works
    """
    parent_a = net.Network(1, 2,
                               [11, 9, 7, 5, 3],
                               ['relu', 'relu', 'relu', 'softmax', 'softmax'],
                               )
    parent_b = net.Network(1, 2,
                               [10, 8, 6],
                               ['relu', 'softmax', 'softmax'],
                               )
    print("---------------------------------------")
    print("---------------------------------------")
    # for i in parent_a.model.get_weights():
    #     print(i)
    #     print("--------------")
    # print(parent_a.model.layers)

    crosser = pop.Population(0, 1, 2)
    child_a, child_b = crosser.cross(parent_a, parent_b)
    print("Child A:\n", child_a)
    print("---------------------------------------")
    print("Child B:\n", child_b)
