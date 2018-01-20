import random
from collections import deque
from multiprocessing import Lock

import gym
import numpy as np
from keras.layers import CuDNNGRU, Dense
from keras.models import Sequential


class Brain:
    def __init__(self, state_size, action_size):
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=state_size, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(CuDNNGRU(32))
        self.model.add(Dense(action_size, activation='linear'))

        self.model.compile(loss='mse', optimizer='Adam')


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_rate_min = 0.01
        self.exploration_decay = 0.995
        self.brain = Brain(self.state_size, self.action_size)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)

        act_values = self.brain.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        pass


class Mario:
    def __init__(self):
        lock = Lock()

        self.sample_batch_size = 32
        self.epochs = 10000
        self.env = gym.make('SuperMarioBros-1-1-v0')
        self.env.configure(lock=lock)

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape
        self.agent = None
