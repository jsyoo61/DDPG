import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class Actor(nn.Module):

    def __init__(self, action_space_shape, observation_space_shape):
        super(Actor, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_input = np.prod(observation_space_shape)
        n_output = np.prod(action_space_shape)

        self.dense1 = nn.Linear(n_input, 400)
        # Relu
        self.dense2 = nn.Linear(self.dense1.out_features, 300)
        # Relu
        self.dense3 = nn.Linear(self.dense2.out_features, n_output)
        # Tanh

    def __add__(self, other):

        dummy = copy.deepcopy(self)

        # Same class
        if type(other) == Actor:
            # Add parameters
            for parameter, parameter1, parameter2 in zip(dummy.parameters(), self.parameters(), other.parameters()):
                parameter.data.copy_(parameter1 + parameter2)

            return dummy

        else:
            # If other is a number
            try:
                # Add parameters
                for parameter, parameter1 in zip(dummy.parameters(), self.parameters()):
                    parameter.data.copy_(parameter1 + other)

                return dummy

            # Illegal operation
            except:
                raise Exception("operands do not match between '%s' and '%s'"%( type(self), type(other) ) )

    def __mul__(self, other):

        dummy = copy.deepcopy(self)

        # Multiply parameters
        for parameter, parameter1 in zip(dummy.parameters(), self.parameters()):
            parameter.data.copy_(parameter1 * other)

        return dummy


    def __rmul__(self, other):
        return self.__mul__(other)

    def forward(self, x):

        # h1, h2, o1
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = torch.tanh(self.dense3(x))

        return x



class Actor_BatchNorm(nn.Module):

    def __init__(self, action_space_shape, observation_space_shape):
        super(Actor_BatchNorm, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_input = np.prod(observation_space_shape)
        n_output = np.prod(action_space_shape)

        self.dense1 = nn.Linear(n_input, 400)
        self.bn1 = nn.BatchNorm1d(self.dense1.out_features)
        # Relu
        self.dense2 = nn.Linear(self.dense1.out_features, 300)
        self.bn2 = nn.BatchNorm1d(self.dense2.out_features)
        # Relu
        self.dense3 = nn.Linear(self.dense2.out_features, n_output)
        # Tanh

    def __add__(self, other):

        dummy = copy.deepcopy(self)

        # Same class
        if type(other) == Actor_BatchNorm:
            # Add parameters
            for parameter, parameter1, parameter2 in zip(dummy.parameters(), self.parameters(), other.parameters()):
                parameter.data.copy_(parameter1 + parameter2)

            return dummy

        else:
            # If other is a number
            try:
                # Add parameters
                for parameter, parameter1 in zip(dummy.parameters(), self.parameters()):
                    parameter.data.copy_(parameter1 + other)

                return dummy

            # Illegal operation
            except:
                raise Exception("operands do not match between '%s' and '%s'"%( type(self), type(other) ) )

    def __mul__(self, other):

        dummy = copy.deepcopy(self)

        # Multiply parameters
        for parameter, parameter1 in zip(dummy.parameters(), self.parameters()):
            parameter.data.copy_(parameter1 * other)

        return dummy


    def __rmul__(self, other):
        return self.__mul__(other)

    def forward(self, x):

        # h1, h2, o1
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))
        x = torch.tanh(self.dense3(x))

        return x

class Critic(nn.Module):

    def __init__(self, action_space_shape, observation_space_shape):
        super(Critic, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_input = np.prod(observation_space_shape)
        n_action = np.prod(action_space_shape)

        self.dense1 = nn.Linear(n_input, 400)
        # self.Relu
        self.dense2 = nn.Linear(self.dense1.out_features, 300)
        # self.Relu
        self.dense3 = nn.Linear(self.dense2.out_features + n_action, 1)
        # self.tanh

    def __add__(self, other):

        dummy = copy.deepcopy(self)

        # Same class
        if type(other) == Critic:
            # Add parameters
            for parameter, parameter1, parameter2 in zip(dummy.parameters(), self.parameters(), other.parameters()):
                parameter.data.copy_(parameter1 + parameter2)

            return dummy

        else:
            # If other is a number
            try:
                # Add parameters
                for parameter, parameter1 in zip(dummy.parameters(), self.parameters()):
                    parameter.data.copy_(parameter1 + other)

                return dummy

            # Illegal operation
            except:
                raise Exception("operands do not match between '%s' and '%s'"%( type(self), type(other) ) )

    def __mul__(self, other):

        dummy = copy.deepcopy(self)

        # Multiply parameters
        for parameter, parameter1 in zip(dummy.parameters(), self.parameters()):
            parameter.data.copy_(parameter1 * other)

        return dummy


    def __rmul__(self, other):
        return self.__mul__(other)

    def forward(self, x, action):

        # h1, h2, o1
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = torch.cat((x, action), dim = -1)
        x = torch.tanh(self.dense3(x))

        return x

class Critic_BatchNorm(nn.Module):

    def __init__(self, action_space_shape, observation_space_shape):
        super(Critic_BatchNorm, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_input = np.prod(observation_space_shape)
        n_action = np.prod(action_space_shape)

        self.dense1 = nn.Linear(n_input, 400)
        self.bn1 = nn.BatchNorm1d(self.dense1.out_features)
        # self.Relu
        self.dense2 = nn.Linear(self.dense1.out_features, 300)
        self.bn2 = nn.BatchNorm1d(self.dense2.out_features)
        # self.Relu
        self.dense3 = nn.Linear(self.dense2.out_features + n_action, 1)
        # self.tanh

    def __add__(self, other):

        dummy = copy.deepcopy(self)

        # Same class
        if type(other) == Critic_BatchNorm:
            # Add parameters
            for parameter, parameter1, parameter2 in zip(dummy.parameters(), self.parameters(), other.parameters()):
                parameter.data.copy_(parameter1 + parameter2)

            return dummy

        else:
            # If other is a number
            try:
                # Add parameters
                for parameter, parameter1 in zip(dummy.parameters(), self.parameters()):
                    parameter.data.copy_(parameter1 + other)

                return dummy

            # Illegal operation
            except:
                raise Exception("operands do not match between '%s' and '%s'"%( type(self), type(other) ) )

    def __mul__(self, other):

        dummy = copy.deepcopy(self)

        # Multiply parameters
        for parameter, parameter1 in zip(dummy.parameters(), self.parameters()):
            parameter.data.copy_(parameter1 * other)

        return dummy


    def __rmul__(self, other):
        return self.__mul__(other)

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
#
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
# %% test on modifying layer weights manually
# observation_space_shape = (2,)
# action_space_shape = (1,)
# a = Actor(action_space_shape = action_space_shape, observation_space_shape = observation_space_shape)
# b = Actor(action_space_shape = action_space_shape, observation_space_shape = observation_space_shape)
# for i in a.parameters():
#     print(i)
# a.
#
# a.dense1.weight
# b.dense1.weight
# 5.1652e-01 + -4.5715e-01
# c = (a.dense1.weight + b.dense1.weight)
# c
# a.dense1.weight.add(b.dense1.weight)
# a.dense1.weight
# a.dense1.weight.add_(b.dense1.weight)
# a.dense1.weight = a.dense1.weight + b.dense1.weight
# a.dense1.weight.copy_(a.dense1.weight + b.dense1.weight)
# a.dense1.weight.data.copy_(a.dense1.weight + b.dense1.weight)
# help(a.dense1.weight)
# a.dense1.weight
# a.dense1.weight = (torch.nn.parameter.Parameter(a.dense1.weight + b.dense1.weight))
#
# a.parameters
# type(a.dense1.weight)
# a.dense1.weight.add()
# type(a.dense1.weight.abs())
#
# torch.add(a.dense1.weight, b.dense1.weight)
# a.dense1.weight.add(b.dense1.weight)
# for i in a.buffers():
#     print(i)
# for i in a.children():
#     print(i)
# a.state_dict().keys()
#
# a.state_dict().keys()
#
# type(a) == Actor
#
# a.dense1.weight
# b.dense1.weight
# c = a.dense1.weight + b.dense1.weight
# c
# (a+b).dense1.weight
# a = a+b
# a.dense1.weight
# (a * 2).dense1.weight
# a = a * 0.9 + 0.1 * b
# (a * 0.9).dense1.weight
# 0.1 * b
# (0.9*a).dense1.weight
#
#
# # %%
# c = Critic(action_space_shape = action_space_shape, observation_space_shape = observation_space_shape)
# action_space_shape
# observation_space_shape
#
# action = torch.tensor([[1],[1]], dtype = torch.float)
# observation = torch.tensor([[3,3],[2.,2.]], dtype = torch.float)
# Q = c(observation, action)
# Q.backward()
# observation.type()
# l1 = nn.Linear(2, 400)
# l2 = nn.Linear(l1.out_features, 300)
# l3 = nn.Linear(l2.out_features, 1)
# observation
# q = l3(l2(l1(observation)))
# q.shape
#
# q_mean = q.mean()
# q_mean.backward()
# observation.grad
# print(observation.grad)
# l1.weight.grad.shape
#
#
# print(torch.device)
# torch.device.
# a = torch.tensor([[1],[1]])
# a.type()
# action.type()
#
# dd
# observation.shape


# %% a

#
# a.parameters()
#
# for p1, p2 in zip(a.parameters(), b.parameters()):
#     p1.data.copy_(p1 + p2)
#
# a.dense1.weight
# a.dense1.weight * 2
#
# for p in a.parameters():
#     p.data.copy_(p * 2)
#
#
# a.bn1.weight + b.bn1.weight
#
# a.double()
# a.eval()
# a.extra_repr()
# a.float()
# a.forward(torch.tensor([1,2,3]))
# a.half()
# a.load_state_dict()
# for i in a.modules():
#     print(i)
# for i in a.named_parameters():
#     print(i)a = Actor()

# raise ValueError('asdf')
# import numpy as np
#
# np.source(np.array)
# np.source(np.ndarray)

'''
buffers?
children?
parameters?

any more generator objects?
'''
