import gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from preprocess import Preprocessor
from model import Agent

# Hyperparameters
game_name = 'MountainCarContinuous-v0'
million = 1000000
n_train_steps = 50 * million

env = gym.make(game_name)
action_space_shape = env.action_space.shape

# Create instance of Agent
agent = Agent(action_space_shape = action_space_shape, n_train_steps = n_train_steps)

n_episode = 0

# Start episodes
while(agent.train_complete is False):
    # Count episodes
    n_episode += 1

    # observation: [2 values]
    observation = env.reset()
    agent.reset(observation)

    done = False
    timestep = 0

    print('Episode start: %s'%(episode))

    # Play game
    while(done is False):
        env.render()

        action = agent.act()
        observation, reward, done, info = env.step(action)
        agent.observe(observation, reward)
        agent.train()

        timestep += 1

    print('Episode finished after timestep: %s'%(timestep))

env.close()
print('Training complete after episode: %s'%(n_episode))



#
# reward = []
# ob = env.reset()
# import time
# for i in range(600):
#     env.render()
#     time.sleep(1/60)
#     ob, re, do, info = env.step(env.action_space.sample())
#     reward.append(re)
#     # info
#
# env.action_space.sample()
