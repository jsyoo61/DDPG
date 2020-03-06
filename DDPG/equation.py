import numpy as np

class OUNoise():
    """docstring for OUNoise"""
    def __init__(self, action_space_shape, mu=0, theta=0.15, sigma=0.2):
        self.action_space_shape = action_space_shape
        self.n_action_space = np.prod(self.action_space_shape)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def __call__(self):
        return self.noise()

    def reset(self):
        self.state = np.full(self.n_action_space, self.mu, dtype = np.float64)

    def noise(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.n_action_space)
        self.state += dx
        return self.state
