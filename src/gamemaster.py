""" Contains the test environment for evaluation """
import gym
from multiprocessing import Pool
from functools import partial

class GameMaster:
    def __init__(self, env_name, thread_num=12):
        self.envs = [gym.make(env_name) for _ in range(thread_num)]
        self.n = thread_num

    def eval_fitnesses(self, networks, time_steps=1000, render=False):
        fitnesses = []
        for batch in range(0, len(networks), self.n):
            pool = Pool(processes=self.n)
            net_stripped = [network.strip_to_components(i) for i, network in enumerate(networks[batch:batch+self.n])]
            results = pool.map(partial(self.eval, time_steps, render, networks[0].input_dim, networks[0].output_dim),
                        net_stripped)
            fitnesses += results
        return fitnesses

    def eval(self, time_steps, render, input_dim, output_dim, specs):
        import network as net
        network = net.Network(input_dim, output_dim, specs[0], specs[1])
        env = self.envs[specs[2]%self.n]

        print("Starting Training Room %d" % specs[2])
        observations = env.reset()
        reward = -1
        for t in range(time_steps):
            if render: env.render()
            action = network.get_category(network.run(observations))
            observation, reward, done, info = env.step(action)
            if done:
                print("Training Room %d finished up at time step %d" % (specs[2], t))
                break
        return reward