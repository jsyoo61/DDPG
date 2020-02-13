import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):

    def __init__(self, action_space_shape, observation_shape):
        super(Actor, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_input = np.prod(observation_shape)
        n_output = np.prod(action_space_shape)

        self.dense1 = nn.Linear(n_input, 400)
        self.bn1 = nn.BatchNorm1d(self.dense1.out_features)
        # self.Relu
        self.dense2 = nn.Linear(self.dense1.out_features, 300)
        self.bn2 = nn.BatchNorm1d(self.dense2.out_features)
        # self.Relu
        self.dense3 = nn.Linear(self.dense2.out_features, n_output)
        # self.tanh

    def forward(self, x):

        # h1, h2, o1
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))
        x = F.tanh(self.dense3(x))

        return x

class Critic(nn.Module):

    def __init__(self, action_space_shape, observation_shape):
        super(Actor, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_input = np.prod(observation_shape)
        n_action = np.prod(action_space_shape)

        self.dense1 = nn.Linear(n_input, 400)
        self.bn1 = nn.BatchNorm1d(self.dense1.out_features)
        # self.Relu
        self.dense2 = nn.Linear(self.dense1.out_features, 300)
        self.bn2 = nn.BatchNorm1d(self.dense2.out_features)
        # self.Relu
        self.dense3 = nn.Linear(self.dense2.out_features + n_action, 1)
        # self.tanh

    def forward(self, x, action):

        # h1, h2, o1
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))
        x = torch.cat((x, action), dim = -1)
        x = torch.tanh(self.dense3(x))

        return x



# class A():
#     pass
# self = A()
# observation_shape = (2,)
# action_space_shape = (1,)
# self.bn1.
# nn.ReLU
#
# a = torch.tensor([[1,2],[3,4]])
# a
# b = torch.tensor([[5,6],[7,8]])
# b
# a+b
# a*b
# a@b
#
# x = np.full((20,2), 1)
# x.shape
# x = torch.from_numpy(x).to(device, dtype=torch.float)
# x
# x.shape
# action = np.full((20,1), 1)
# action.shape
# action = torch.from_numpy(action).to(device, dtype=torch.float)
# y
# y.shape
# self.dense3.in_features
# z = F.tanh(self.dense3(y))
# help(nn.Module)
#
# -0.9732 + 10 * np.array([0.3743,-1.7724,-0.5811,-0.8017])
#
# for i in a.parameters():
#     print(i)
# a.dense1.weight
# a.dense1.weight + b.dense1.weight
# a.dense1.weight = a.dense1.weight + b.dense1.weight
# a.dense1.weight.data.copy_(a.dense1.weight + b.dense1.weight)
# type(a.dense1.weight)
# a.dense1.weight.add()
# type(a.dense1.weight.abs())
#
# torch.add(a.dense1.weight, b.dense1.weight)
# a.dense1.weight.add(b.dense1.weight)
#
# for i in a.children():
#     print(i)
# a.state_dict().keys()
# a.modules + b.modules
