import random
import numpy as np
from brain import Brain

MAX_EPSILON = 1
MIN_EPSILON = 0.1

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, state_cnt, action_cnt):
        self.state_cnt = state_cnt
        self.action_cnt = action_cnt

        self.brain = Brain(state_cnt, action_cnt)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_cnt - 1)
        else:
            return numpy.argmax(self.brain.predict_one(s))