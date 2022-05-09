import gym
from gym.spaces import Discrete, Box
import random
import numpy as np

class MyEnvironment(gym.Env):

    def __init__(self, env_config=None):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=-1., high=1., shape=(2,), dtype=np.float64)
        self.current_context = None
        print(env_config)
        self.rewards_for_context = {
            -1. : [-10, 0, 10],
            1. : [10, 0, -10]
        }


    def reset(self):
        self.current_context = random.choice([-1, 1])
        return np.array([-self.current_context, self.current_context])

    def step(self, action):
        reward = self.rewards_for_context[self.current_context][action]
        return (np.array([-self.current_context, self.current_context]), reward, True, {"regret":10 - reward})

    def __repr__(self) -> str:
        return f"(SimpleContextualBandit(action={self.action_space}, observation_space={self.observation_space}, current_context={self.current_context}, reward_per_context={self.rewards_for_context})"



