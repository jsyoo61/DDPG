import gym
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import deque
from preprocess import Preprocessor
from agent import Agent
from tools import save_pickle

# Hyperparameters
game_name = 'MountainCarContinuous-v0'
million = 1000000
n_train_steps = 50 * million

env = gym.make(game_name)
action_space_shape = env.action_space.shape
print(env.action_space.low, env.action_space.high)
observation_space_shape = env.observation_space.shape

# Create instance of Agent
agent = Agent(action_space_shape = action_space_shape, observation_space_shape = observation_space_shape, n_train_steps = n_train_steps)

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

    print('Episode start: %s'%(n_episode))

    # Play game
    while(done is False and agent.train_complete is False):
        env.render()

        action = agent.act()
        observation, reward, done, info = env.step(action)
        agent.observe(observation, reward)
        agent.train()

        timestep += 1
        print('\rtime_step:%s, train_step:%s, action:%s, reward:%s '%(agent.time_step, agent.train_step, action, reward), end = '', flush = True)

    print('Episode finished after timestep: %s'%(timestep))

env.close()
print('Training complete after episode: %s'%(n_episode))
save_pickle('agent.p', agent)

#
# action
# agent.time_step
# agent.train_step
# agent.action.
# agent.previous_observation.view((6,)).shape
# agent.previous_observation.shape
# agent.observation_input_shape
# agent.k
# observation
# agent.action_space_shape * agent.k
# agent.actor.forward(agent.previous_observation) - 0.05482081
# agent.actor.forward(agent.previous_observation, requires_grad = False)
# agent.action
# agent.action.numpy()
# agent.observation_input_shape
# (-1,) + agent.observation_input_shape
#
# a=(1,)
# a
# b=[1,2,3]
# list(a)
# b
# tuple(b)
# from tools import multiply_tuple
#
# tuple([[2,3],[4]])
# multiply_tuple(a, 3)
#
# a = (1,2,3)
# num = 3
# mul_tu = list()
# for e in a:
#     mul_tu.append(e*num)
#
# mul_tu
# c = tuple(mul_tu)
# c
# mul_tu = tuple(mul_tu)
# (1,2) + (3,)
# a.append(2)
#
#
# import torch
# import torch.nn.functional as F
# memory_size = len(agent.replay_memory)
#
# # state, action, reward, state_next
# s_i = list()
# a_i = list()
# r_i = list()
# s_i_1 = list()
#
# # Random Sample transitions, append them into np arrays
# random_index = np.random.choice(memory_size, size = agent.minibatch_size, replace = False) # random_index: [0,5,4,9, ...] // "replace = False" makes the indices exclusive.
#
# for index in random_index:
#
#     # Random sample transitions, 'minibatch' times
#     s, a, r, s_1 = agent.replay_memory[index]
#     s_i.append(s) # s_i Equivalent to [self.replay_memory[index][0] for index in random_index]
#     a_i.append(a)
#     r_i.append(r)
#     s_i_1.append(s_1)
# torch.as_tensor(r_i)
# torch.tensor(r_i)
# r_i[0]
# s_i[0].shape
# torch.stack(r_i)
# s_i = torch.as_tensor(s_i, dtype = torch.float, device = agent.device)
# s_i_new = torch.stack(s_i).to(dtype = torch.float, device = agent.device)
# s_i_new.shape
#
# s_i, a_i, r_i, s_i_1 = agent.random_sample_data() # **minibatch info included in "agent"
#
# # 2. Set y_i
# y_i = r_i + agent.gamma * agent.critic_hat.forward( s_i_1, agent.actor_hat.forward(s_i_1) )
#
# # 3. Calculate Loss
# agent.optimizer_critic.zero_grad()
# critic_loss = F.mse_loss( y_i, agent.critic.forward( s_i, a_i ) )
#
# # 4. Update Critic
# critic_loss.backward()
# agent.optimizer_critic.step()
# critic_loss
# y_i.grad
# agent.optimizer_critic.param_groups[0]['params'][0].shape
# agent.optimizer_critic.param_groups[0]['params'][1].shape
# agent.optimizer_critic.param_groups[0]['params'][2].shape
# agent.optimizer_critic.param_groups[0]['params'][3].shape
# agent.optimizer_critic.param_groups[0]['params'][4].shape
# agent.optimizer_critic.param_groups[0]['params'][5].shape
#
# agent.optimizer_critic.param_groups[0]['params'][0].grad
# s_i.requires_grad
# a_i.requires_grad
# a_i
# r_i.requires_grad
# s_i_1.requires_grad
# agent.optimizer_critic.param_groups[0]['params'][0].requires_grad
#
#
# a = torch.full((3,1),1)
# a
# b = torch.full((3,), 3)
# a.shape
# b.shape
# a
# b
# a+b
# d = 5
# torch.as_tensor([d])
# l=list()
# for i in range(5):
#     l.append(torch.tensor([d]))
# l
# torch.stack(l).shape
# ab = torch.tensor([0])
# ab += 1
# ab
