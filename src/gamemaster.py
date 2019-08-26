""" Contains the test environment for evaluation """
import gym
import multiprocessing_gym as mpg

class GameMaster:
    def __init__(self, env_name, thread_count=64):
        self.envs = self.make_mp_envs(env_name, thread_count, 2)
        self.batch_size = thread_count

    def make_mp_envs(env_id, num_env, seed, start_idx=0):
        def make_env(rank):
            def fn():
                env = gym.make(env_id)
                env.seed(seed + rank)
                return env
            return fn
        return mpg.SubprocVecEnv([make_env(i + start_idx) for i in range(num_env)])

    def eval_fitness(self, networks, time_steps=1000, render=True):
        print("Evaluating Fitness...")
        if len(networks) < self.batch_size:
            print("Env Training Warning: Amount of Networks smaller than env number, could be dangerous")
        print("Total of %d Eval Batches" % (len(networks)//self.batch_size)+1)
        fitnesses = []
        for batch in range(0, len(networks), self.batch_size):
            observations = self.envs.reset()
            rewards = [0]*self.batch_size
            for t in range(time_steps):
                if render: self.envs.render()
                action_list = []
                for _ in range(batch, batch+self.batch_size):
                    network = networks[_]
                    action_list.append(network.get_category(network.run(observations[_-batch])))
                observations, rewards, dones, infos = self.envs.step(action_list)
            fitnesses.append(rewards)
        self.envs.close()
        return fitnesses