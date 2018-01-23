import random
from collections import deque
from multiprocessing import Lock

import gym
import numpy as np
from keras.layers import CuDNNGRU, Dense, Reshape, Conv2D, Permute
from keras.models import Sequential


class Brain:
    def __init__(self, state_size, action_size):
        self.model = Sequential()
        # self.model.add(Dense(32, input_shape=state_size, activation='relu'))
        self.model.add(Conv2D(32, (8, 8), strides=(4,4), activation='relu', input_shape=state_size))
        self.model.add(Conv2D(64, (4, 4), strides=(2,2), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), strides=(1,1), activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        #self.model.add(CuDNNGRU(32))
        self.model.add(Dense(action_size, activation='linear'))
        # self.model.add(Reshape((-1, action_size)))

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
        # if np.random.rand() <= self.exploration_rate:
        #     return np.random.choice([0,1], self.action_size)

        # act_values = self.brain.model.predict(state)
        # action = np.argmax(act_values[0][0])
        act_values = self.brain.model.predict(state)
        print(act_values.shape)
        action = np.argmax(act_values)
        print(action)
        action_tensor = np.zeros(self.action_size)
        action_tensor[action] = 1

        return action_tensor

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return

        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.brain.model.predict(next_state)[0])
            target_f = self.brain.model.predict(state)
            target_f[0][action] = target
            self.brain.model.fit(state, target_f, epochs=1, verbose=0)

        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_decay


class Mario:
    def __init__(self):
        self.sample_batch_size = 32
        self.epochs = 10000
        self.env = gym.make('SuperMarioBros-1-1-v0')

        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.shape
        self.agent = Agent(self.state_size, self.action_size)

    def run(self):
        try:
            for index_epoch in range(self.epochs):
                state = self.env.reset()
                state = np.expand_dims(state, axis=0)

                done = False
                index = 0
                while not done:
                    self.env.render()

                    action = self.agent.act(state)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.expand_dims(next_state, axis=0)
                    self.agent.remember(
                        state, action, reward, next_state, done)
                    state = next_state
                    index += 1

                print("Episode {}# Score: {}".format(index_epoch, index + 1))
                self.agent.replay(self.sample_batch_size)
        finally:
            pass


if __name__ == '__main__':
    mario = Mario()
    mario.run()
