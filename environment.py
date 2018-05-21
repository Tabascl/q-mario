import numpy as np
import gym_super_mario_bros
from util import process_image
from agent import Agent
from random_agent import RandomAgent

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
IMAGE_STACK = 2

MEMORY_CAPACITY = 10000

class Environment:
    def __init__(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')

    def run(self, agent):
        img = self.env.reset()
        w = process_image(img)
        s = np.array([w, w])

        R = 0
        while True:
            a = agent.act(s)

            r = 0
            img, r, done, info = self.env.step(a)
            s_ = np.array([s[1], process_image(img)])

            r = np.clip(r, -1, 1)

            if done:
                s_ = None

            agent.observe((s, a, r, s_))
            agent.replay()

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)

env = Environment()

state_cnt = (IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)
action_cnt = env.env.action_space.n

agent = Agent(state_cnt, action_cnt, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)
random_agent = RandomAgent(action_cnt)

try:
    print("Initialize with random agent...")
    while random_agent.exp < MEMORY_CAPACITY:
        env.run(random_agent)
        print(random_agent.exp, "/", MEMORY_CAPACITY)

    agent.memory = random_agent.memory

    random_agent = None

    print("Start learning")
    while True:
        env.run(agent)
finally:
    agent.brain.model.save("Mario-DQN-PER.h5")