import numpy as np
import copy
from collections import deque
from preprocess import Preprocessor
from model import Actor, Critic


class Agent():

    def __init__(self, action_space, n_train_steps = 50 * 1000000, replay_memory_size = 1000000, k = 3):

        # Hyperparameters - dynamic
        self.action_space = action_space
        self.n_train_steps = n_train_steps
        self.replay_memory_size = replay_memory_size
        self.replay_memory = deque(maxlen = self.replay_memory_size)
        self.k = k
        self.tau = 0.001

        # Hyperparameters - static
        self.minibatch_size = 32
        self.gamma = 0.99
        self.update_frequency

        self.replay_start_frame = 50000

        # Parameters - etc
        self.action = None
        self.timestep = 0
        self.train_step = 0
        self.train_complete = False

        # Modules
        self.actor = Actor()
        self.critic = Critic()
        self.actor_hat = copy.deepcopy(self.actor)
        self.critic_hat = copy.deepcopy(self.critic)

        # Operations
        self.mode('train')

    def reset(self, observation):

        self.previous_observation = np.asarray( [observation] * self.k )
        self.observation_buffer = list()
        self.reward = 0

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
        if self.timestep % self.k == 0:
            return True
        else:
            return False

    def act(self):

        if self.wakeup() == True:
            self.action = self.actor.forward(self.previous_observation)

        self.timestep += 1

        return self.action

    def observe(self, observation, reward):

        if self.wakeup() == True:

            # Append observation
            self.observation_buffer.append(observation)
            self.new_observation = np.asarray(self.observation_buffer)

            # Add reward
            self.reward += reward

            # Store transition in replay memory
            # If memory size exceeds, the oldest memory is popped (deque property)
            self.replay_memory.append( (self.previous_observation, self.action, self.reward, self.new_observation) )

            # The new observation will be the previous observation next time
            self.previous_observation = self.new_observation

            # Empty observation buffer, reset reward
            self.observation_buffer = list()
            self.reward = 0

        else:

            self.observation_buffer.append(observation)
            self.reward += reward

    def train(self):

        if self.wakeup() == True:
            # 1. Sample random minibatch of transitions from replay memory
            s_i, a_i, r_i, s_i_1 = random sample

            # 2. Set y_i
            y_i = r_i + self.gamma * self.critic_hat.forward( s_i_1, self.actor_hat.forward(s_i_1) )

            # 3. Calculate Loss
            Loss = torch.reduce_mean( torch.square(y_i - self.critic.forward( s_i, a_i ) )  )

            # 4. Update Critic


            # 5. Update Actor


            # 6. Update target networks
            self.critic_hat = self.tau * self.critic + (1 - self.tau) * self.critic_hat
            self.actor = self.tau * self.actor + (1 - self.tau) * self.actor_hat

            # 7. Increment train step.
            # If train step meets its scheduled training steps, change "train_complete" status
            self.train_step += 1
            if self.train_step >= self.n_train_steps:
                self.train_complete = True
