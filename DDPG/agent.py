import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from collections import deque
from preprocess import Preprocessor
from model import Actor, Critic
from equation import OUNoise
from tools import multiply_tuple

class Agent():

    def __init__(self, action_space_shape, observation_space_shape, n_train_steps = 50 * 1000000, replay_memory_size = 1000000, k = 3):

        # Cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters - dynamic
        self.action_space_shape = action_space_shape
        self.observation_space_shape = observation_space_shape
        self.k = k
        self.observation_input_shape = multiply_tuple(self.observation_space_shape, self.k)
        self.n_train_steps = n_train_steps
        self.replay_memory_size = replay_memory_size
        self.replay_memory = deque(maxlen = self.replay_memory_size)

        # Hyperparameters - static
        self.training_start_time_step = 1000 # Minimum: k * minibatch_size == 3 * 64 = 192
        self.gamma = 0.99 # For reward discount
        self.tau = 0.001 # For soft update

        # Hyperparameters - Ornstein_Uhlenbeck_noise
        self.theta = 0.15
        self.sigma = 0.2
        self.Ornstein_Uhlenbeck_noise = OUNoise(action_space_shape = self.action_space_shape, theta = self.theta, sigma = self.sigma)

        # Hyperparameters - NN model
        self.minibatch_size = 64 # For training NN
        self.lr_actor = 10e-4
        self.lr_critic = 10e-3
        self.weight_decay_critic = 10e-2

        # Parameters - etc
        self.action = None
        self.time_step = 0
        self.train_step = 0
        self.train_complete = False

        # Modules
        self.actor = Actor(action_space_shape = self.action_space_shape, observation_space_shape = self.observation_input_shape).to(self.device)
        self.critic = Critic(action_space_shape = self.action_space_shape, observation_space_shape = self.observation_input_shape).to(self.device)
        self.actor_hat = copy.deepcopy(self.actor)
        self.critic_hat = copy.deepcopy(self.critic)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr = self.lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr = self.lr_critic, weight_decay = self.weight_decay_critic)

        # Operations
        self.mode('train')

    def reset(self, observation):

        self.previous_observation = torch.tensor( [observation] * self.k).to(dtype = torch.float, device = self.device).view(self.observation_input_shape)
        self.observation_buffer = list()
        self.reward = torch.tensor([0]) # Tensor form for compatibility
        self.Ornstein_Uhlenbeck_noise.reset()

        # Since replay memory is somewhat full, we can decrease waiting time for sufficient data to fill in the replay memory.
        self.training_start_time_step = max(0, self.training_start_time_step - self.time_step)
        self.time_step = 0
        # Don't reset replay_memory
        # self.replay_memory = deque(maxlen = self.replay_memory_size)

    def mode(self, mode):

        self.mode = mode
        if self.mode == 'train':
            pass
        elif self.mode == 'test':
            pass
        else:
            assert False, 'mode not specified'

    def wakeup(self):

        # Frame skipping
        # See & Select actions every kth frame. Modify ations every kth frame
        # Otherwise, skip frame
        if self.time_step % self.k == 0:
            return True
        else:
            return False

    def act(self):

        if self.wakeup() == True:
            self.action = self.actor.forward(self.previous_observation) + torch.as_tensor(self.Ornstein_Uhlenbeck_noise(), dtype = torch.float, device = self.device)

        self.time_step += 1

        # Return numpy version
        return self.action.detach().numpy()

    def observe(self, observation, reward):

        if self.wakeup() == True:

            # Append observation
            self.observation_buffer.append(observation)
            self.new_observation = torch.tensor(self.observation_buffer).to(dtype = torch.float, device = self.device).view(self.observation_input_shape)

            # Add reward
            self.reward += reward

            # Store transition in replay memory
            # If memory size exceeds, the oldest memory is popped (deque property)
            # wrap self.action with torch.tensor() to reset requires_grad = False
            self.replay_memory.append( (self.previous_observation, self.action.clone().detach(), self.reward, self.new_observation) ) # self.action.new_tensor() == self.action.clone().detach()

            # The new observation will be the previous observation next time
            self.previous_observation = self.new_observation

            # Empty observation buffer, reset reward
            self.observation_buffer = list()
            self.reward = torch.tensor([0]) # Tensor form for compatibility

        else:

            self.observation_buffer.append(observation)
            self.reward += reward

    def random_sample_data(self):

        memory_size = len(self.replay_memory)

        # state, action, reward, state_next
        s_i = list()
        a_i = list()
        r_i = list()
        s_i_1 = list()

        # Random Sample transitions, append them into np arrays
        random_index = np.random.choice(memory_size, size = self.minibatch_size, replace = False) # random_index: [0,5,4,9, ...] // "replace = False" makes the indices exclusive.

        for index in random_index:

            # Random sample transitions, 'minibatch' times
            s, a, r, s_1 = self.replay_memory[index]
            s_i.append(s) # s_i Equivalent to [self.replay_memory[index][0] for index in random_index]
            a_i.append(a)
            r_i.append(r)
            s_i_1.append(s_1)

        s_i = torch.stack(s_i).to(dtype = torch.float, device = self.device)
        a_i = torch.stack(a_i).to(dtype = torch.float, device = self.device)
        r_i = torch.stack(r_i).to(dtype = torch.float, device = self.device)
        s_i_1 = torch.stack(s_i_1).to(dtype = torch.float, device = self.device)

        return s_i, a_i, r_i, s_i_1

    def train(self):

        if self.wakeup() == True and self.time_step >= self.training_start_time_step:
            # 1. Sample random minibatch of transitions from replay memory
            # state, action, reward, state_next
            s_i, a_i, r_i, s_i_1 = self.random_sample_data() # **minibatch info included in "self"

            # 2. Set y_i
            y_i = r_i + self.gamma * self.critic_hat.forward( s_i_1, self.actor_hat.forward(s_i_1) )

            # 3. Calculate Loss
            self.optimizer_critic.zero_grad()
            critic_loss = F.mse_loss( y_i, self.critic.forward( s_i, a_i ) )

            # 4. Update Critic
            critic_loss.backward()
            self.optimizer_critic.step()

            # 5. Update Actor
            self.optimizer_actor.zero_grad()
            critic_Q_mean =  - self.critic.forward( s_i, self.actor.forward( s_i ) ).mean()
            critic_Q_mean.backward()
            self.optimizer_actor.step()

            # 6. Update target networks
            self.critic_hat = self.tau * self.critic + (1 - self.tau) * self.critic_hat
            self.actor = self.tau * self.actor + (1 - self.tau) * self.actor_hat

            # 7. Increment train step.
            # If train step meets its scheduled training steps, change "train_complete" status
            self.train_step += 1
            if self.train_step >= self.n_train_steps:
                self.train_complete = True

#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# help(torch.functional)
# F.mse_loss(torch.tensor([1,2,3]), torch.tensor([0,0,0]))
# a = torch.tensor([1,2,3])
# b = torch.tensor([0,0,0])
# a = a.float()
# b = b.float()
#
# output1 = F.mse_loss(a,b)
# loss = nn.MSELoss()
# output2 = loss(a,b)
#
# output1.backward()
# output2.backward()
#
# type(output1)
# type(output2)
# type(loss)
# type(F.mse_loss)
# help(nn.MSELoss)
#
# a.int()
# b.int()
# a.dtype
# b.dtype
# a.to(torch.float)
# a.dtype
# type(a)
#
# help(torch.device)
# device = torch.device('cpu')
# torch.cuda.is_available()
#
# class A(nn.Module):
#
#     def __init__(self):
#         super(A, self).__init__()
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         n_input = 5
#         n_output = 3
#
#         self.dense1 = nn.Linear(n_input, 400)
#         self.dense2 = nn.Linear(self.dense1.out_features, 300)
#         self.dense3 = nn.Linear(self.dense2.out_features, n_output)
#
#     def forward(self, x):
#
#         x = self.dense1(x)
#         x = self.dense2(x)
#         x = self.dense3(x)
#
#         return x
#
# model = A()
#
# model
# data = torch.full((10,5),2)
# target = torch.ones(10,3)
# help(torch.rand)
# torch.rand(10,5)
#
# y = model(data)
# y.shape
# y
# loss = F.mse_loss(y, target)
# loss
# loss.
# loss.backward()
# model.dense1.zero_grad()
# data
# model.zero_grad()
# model.dense1.weight.grad
# print(data.grad)
