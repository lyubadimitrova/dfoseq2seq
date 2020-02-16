import torch

class GradEstimator:
    """
    Contains three methods for gradient estimation, each instantiated GradEstimator object uses one of them.
    For math, see https://arxiv.org/pdf/1804.02395.pdf.

    The first method is the 'vanilla' gradient estimator, based on a standard REINFORCE score function estimator.
    The other two are based on the finite differences method for numerical differentiation, 
    namely forward finite differences, and central differences.

    """
    def __init__(self, est_type, sigma, use_only_best, normalizer='sigma'):
        """

        :param est_type:
        """
        self.est_type = est_type
        if self.est_type == 'vanilla':
            self.estimate = self.estimate_grad_vanilla
        elif self.est_type == 'forward':
            self.estimate = self.estimate_grad_forward_fd
        elif self.est_type == 'antithetic':
            self.estimate = self.estimate_grad_antithetic
        else:
            raise ValueError("{} is not a valid gradient estimator.".format(self.est_type))

        self.sigma = sigma
        self.norm_type = normalizer
        self.use_only_best = use_only_best
        self.rewards_stdev = []


    def get_theta_candidates(self, theta, sigma, expl_directions):
        """
        For each estimator, collects the noisy theta candidates needed for the gradient estimation. 
             - Vanilla: theta + sigma*epsilon
             - Antithetic: theta + sigma*epsilon and theta - sigma*epsilon
             - Forward FD: theta + sigma*epsilon and (one) theta 
        """
        candidates = theta + expl_directions * sigma
        if self.est_type == 'antithetic':
            return torch.cat((candidates, theta - expl_directions * sigma))
        if self.est_type == 'forward':
            return torch.cat((candidates, torch.unsqueeze(theta, dim=0).double()))
        return candidates


    def normalizer(self, array):
        """
        Returns a normalizing number, either the standard deviation of the array, or sigma.

        :param array: An array to calculate the standard deviation of. Redundant if using sigma scaling.
        """
        if self.norm_type == 'stdev':
            stdev = array.std()
            self.rewards_stdev.append(stdev)
            return stdev
        return self.sigma


    def estimate_grad_vanilla(self, expl_directions, rewards):
        """
        Standard REINFORCE gradient estimator, with optional scaling by the stdev of the rewards.

        :param expl_directions: The set of perturbations.
        :param rewards_pos: Rewards obtained from perturbed models. rewards_pos[i] is the rewards of 
                            the model perturbed with expl_directions[i].
        """
        if self.use_only_best:
            sort_idx = torch.argsort(rewards, descending=True)
            rewards = rewards[sort_idx][:self.use_only_best]
            expl_directions = expl_directions[sort_idx][self.use_only_best]

        return rewards @ expl_directions / (len(rewards) * self.normalizer(rewards))


    def estimate_grad_forward_fd(self, expl_directions, rewards):
        """
        Forward finite differences gradient estimator, with optional scaling by the stdev of the rewards.

        :param expl_directions: The set of perturbations.
        :param rewards: Rewards obtained from perturbed models: a tensor of size (num_expl_directions + 1).
                        The last element of rewards is the 'baseline', i.e. the reward of the unperturbed
                        model. The rest are rewards of perturbations +epsilon.
        """
        rewards_pos = rewards[:-1]

        if self.use_only_best:
            sort_idx = torch.argsort(rewards_pos, descending=True)
            rewards_pos = rewards_pos[sort_idx][:self.use_only_best]
            expl_directions = expl_directions[sort_idx][:self.use_only_best]

        return (rewards_pos - rewards[-1]) @ expl_directions / (len(rewards_pos) * self.normalizer(rewards_pos))


    def estimate_grad_antithetic(self, expl_directions, rewards):
        """
        Central differences (also: antithetic) gradient estimator, with optional scaling by the stdev of the rewards.

        :param expl_directions: The set of perturbations.
        :param rewards: Rewards obtained from perturbed models, a tensor of size (num_expl_directions * 2), 
                        where the first half corresponds to exploration directions +epsilon, and the second half
                        to -epsilon.
        """
        rewards_pos, rewards_neg = torch.chunk(rewards, 2)   # split the two halves
        rewards = rewards.view((2,-1))

        if self.use_only_best:
            sort_idx = torch.argsort(torch.max(rewards, dim=0).values, descending=True)
            rewards_pos = rewards_pos[sort_idx][:self.use_only_best]
            rewards_neg = rewards_neg[sort_idx][:self.use_only_best] 
            expl_directions = expl_directions[sort_idx][:self.use_only_best]
        
        return (rewards_pos - rewards_neg) @ expl_directions / (len(rewards_pos) * 2 * self.normalizer(torch.cat((rewards_pos, rewards_neg))))
