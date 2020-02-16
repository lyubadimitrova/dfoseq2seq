"""
Optimizers for custom gradients. 
 Adapted to pytorch from https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py
 LICENSE: https://github.com/openai/evolution-strategies-starter/blob/master/LICENSE
"""


import numpy as np
import torch


class Optimizer(object):
    """
    Base optimizer class.
    """
    def __init__(self, dim):
        """
        :param dim: The number of dimensions of theta.
        """
        self.dim = dim
        self.t = 0    # keeps track of the number of optimization steps

    def update(self, theta, grad):
        """
        Updates theta with a step computed with a given gradient.

        :param theta: the parameters to update
        :param grad: the gradient that should be used for the update
        """
        self.t += 1
        step = self._compute_step(grad)
        theta.add_(step)  # in-place adding, no need to return theta
        return step

    def _compute_step(self, grad):
        """
        Implemented in the child classes.

        """
        raise NotImplementedError
        
    def get_state(self):
        """
        Returns all the optimizer attrubutes; differs depending on the child class.
        """
        return vars(self)

    def load_state(self, state_dict):
        """
        Loads optimizer attributes, for example from a DFO checkpoint.

        :param state_dict: a dict like self.get_state() returns
        """
        [setattr(self, attr, value) for attr, value in state_dict.items()]

    def cudafy(self):
        """
        Moves all attributes represented by torch tensors to CUDA.
        """
        for attr, value in self.get_state().items():
            try:
                setattr(self, attr, value.cuda())
            except AttributeError:
                pass



class SGD(Optimizer):
    """
    SGD optimizer with optional constant decay rate. 
    """

    def __init__(self, dim, value_dict):
        """
        :param dim: the dimensionality of theta
        :param value_dict: a dictionary with hyperparameters
        """
        Optimizer.__init__(self, dim)
        self.stepsize = value_dict['start']
        self.decay = value_dict.get('decay', 1.0)
        self.min_stepsize = value_dict.get('min', 0.0001)

    def _compute_step(self, grad):
        """
        Computes one SGD step based on the given gradient.

        :param grad: the gradient to optimize with
        :return: the step, size [dim]
        """
        step = -self.stepsize * grad
        if self.stepsize > self.min_stepsize:
            self.stepsize *= self.decay
        return step


class Momentum(Optimizer):

    def __init__(self, dim, value_dict):
        """
        :param dim: the dimensionality of theta
        :param value_dict: a dictionary with hyperparameters
        """
        Optimizer.__init__(self, dim)
        self.v = torch.zeros(self.dim)
        self.stepsize = value_dict['start']
        self.momentum = value_dict.get('momentum', 0.9)

    def _compute_step(self, grad):
        """
        Computes one Momentum-SGD step based on the given gradient.
        
        :param grad: the gradient to optimize with
        :return: the step, size [dim]
        """
        self.v = self.momentum * self.v + (1. - self.momentum) * grad
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):

    def __init__(self, dim, value_dict):
        """
        :param dim: the dimensionality of theta
        :param value_dict: a dictionary with hyperparameters
        """
        Optimizer.__init__(self, dim)
        self.stepsize = value_dict['start']
        self.beta1 = value_dict.get('beta1', 0.9)
        self.beta2 = value_dict.get('beta2', 0.999)
        self.epsilon = value_dict.get('epsilon', 1e-08)
        self.m = torch.zeros(self.dim)
        self.v = torch.zeros(self.dim)

    def _compute_step(self, grad):
        """
        Computes one Adam step based on the given gradient.
        
        :param grad: the gradient to optimize with
        :return: the step, size [dim]
        """
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        step = -a * self.m / (torch.sqrt(self.v) + self.epsilon)
        return step
