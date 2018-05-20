import random
from collections import deque
import numpy as np
from brain import Brain

MAX_EPSILON = 1
MIN_EPSILON = 0.1

GAMMA = 0.99

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, state_cnt, action_cnt, img_stack, img_width, img_height):
        self.state_cnt = state_cnt
        self.action_cnt = action_cnt
        self.img_stack = img_stack
        self.img_width = img_width
        self.img_height = img_height

        self.memory = deque(maxlen=200000)

        self.brain = Brain(state_cnt, action_cnt, img_stack, img_width,
                           img_height)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_cnt - 1)
        else:
            return np.argmax(self.brain.predict_one(s))

    def observe(self, sample):
        x, y, errors = self._get_targets([(0, sample)])
        self.memory.append((errors[0], sample))

    def _get_targets(self, batch):
        no_state = np.zeros(self.state_cnt)

        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])

        p = self.brain.predict(states)

        p_ = self.brain.predict(states_)
        p_target_ = self.brain.predict(states_, target=True)

        x = np.zeros((len(batch), self.img_stack, self.img_width, self.img_height))
        y = np.zeros((len(batch), self.action_cnt))
        errors = np.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i]
            old_val = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * p_target_[i][ np.argmax(p_[i]) ] # DDQN
            
            x[i] = s
            y[i] = t
            errors[i] = abs(old_val - t[a])
        
        return (x, y, errors)