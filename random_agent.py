from memory import Memory
from numpy import random

class RandomAgent:
    memory = Memory(200000)
    exp = 0

    def __init__(self, action_cnt):
        self.action_cnt = action_cnt

    def act(self, s):
        return random.randint(0, self.action_cnt-1)

    def observe(self, sample):
        error = abs(sample[2])
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass