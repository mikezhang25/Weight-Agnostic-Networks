
# """ The main driver function for population testing and simulation  "

import population as pop
import gamemaster as gm
import argparse

parser = argparse.ArgumentParser(description='Evolves the optimal network structure for a gym environmenmt')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print args.accumulate(args.integers)

if __name__ == '__main__':
    mp = gm.GameMaster('MountainCarContinuous-v0', thread_num=8)
    crowd = pop.Population(128, 1, 1, evaluator=mp)
    crowd.evolve(10, save_dir='./test-1')
