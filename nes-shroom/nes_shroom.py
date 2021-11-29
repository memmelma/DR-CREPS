import os
import sys
import torch
import numpy as np

from copy import deepcopy

from .utils import *

import joblib

class CoreNES():
    """
    :ivar int gen: Current generation
    :ivar Policy policy: Trained policy
    :ivar torch.optim.Optimizer optim: Optimizer of the policy
    """
    def __init__(self, policy, mdp, optimizer=torch.optim.Adam, optimizer_lr=0.02, 
                    n_step=300, n_rollout=2, seed=42):


        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.env = mdp
        self.horizon = mdp.info.horizon

        self.gen = 0
        self.n_rollout = n_rollout
        self.policy = policy
        self.optimizer_lr = optimizer_lr
        self.optim = optimizer(self.policy.parameters(), lr=self.optimizer_lr)
        
        self.dummy_policy = deepcopy(self.policy)

        self.n_step = n_step

        self.best_reward = -np.inf

        self.rewards_mean = []

    def evaluate_populations(self, population_params):
        rewards = []
        reward_array = np.zeros(self.policy.population_size, dtype=np.float32)
        
        for param in population_params:
            self.dummy_policy.set_weights(param)
            rewards.append(self.evaluate_policy(self.dummy_policy))
        
        reward_array[...] = rewards
        return reward_array

    def evaluate_policy(self, policy):
        total_reward = 0
        for _ in range(self.n_rollout):
            done = False
            obs = self.env.reset()
            policy.set_weights(torch.nn.utils.parameters_to_vector(policy.parameters()))

            horizon_counter = 0
            while not done:
                action = policy.draw_action(obs)
                obs, reward, done, _ = self.env.step(action.numpy())
                
                if self.horizon is not None:
                    horizon_counter += 1
                    if horizon_counter == self.horizon:
                        done = True

                total_reward += reward
        return total_reward / self.n_rollout

    def optimize(self, grad, limit_grad=False):
        index = 0
        for parameter in self.policy.parameters():
            size = np.prod(parameter.shape)
            parameter.grad = grad[index:index+size].view(parameter.shape)
            # Limit gradient update to increase stability.
            if limit_grad:
                parameter.grad.data.clamp_(-1.0, 1.0)
            index += size
        self.optim.step()

    def train(self, strat='nes'):
        """Train ``self.policy`` for ``self.config.nes.n_steps`` to increase reward returns
        from the ``self.env`` using Natural Evolution Strategy gradient estimation."""
        torch.set_grad_enabled(False)
     
        n_samples = 0
        for gen in range(self.n_step):
            self.gen = gen

            # Sample
            population_params, epsilons = self.policy.sample()

            # Evaluate Population
            rewards = self.evaluate_populations(population_params)

            if strat == 'nes':
                # Calculate Gradients
                grad = self.policy.calculate_gradients(rewards, epsilons)
                self.optimize(grad)

            elif strat == 'es':
                self.policy.no_grad_update(rewards, population_params, self.optim)
            else:
                return

            n_samples += self.policy.population_size*self.n_rollout
            print(f'samples total: {n_samples}')
            
            reward = self.evaluate_policy(self.policy)

            self.rewards_mean += [[reward]]
            
            if reward > self.best_reward:
                self.best_reward = reward
            print(f'Gen: {self.gen} Test Reward: {reward} Best Reward: {self.best_reward}')

        sys.stdout = sys.__stdout__
        torch.set_grad_enabled(True)

    def log(self, alg_name='NES', seed=42, results_dir='results'):
        
        os.makedirs(results_dir, exist_ok=True)

        self.init_params = dict({
            'n_epochs': self.n_step,
            'fit_per_epoch': self.policy.population_size * self.n_rollout,
            'ep_per_fit': 1
        })

        dump_dict = dict({
            'returns_mean': self.rewards_mean,
            # 'returns_std': returns_std,
            'best_reward': self.best_reward,
            'init_params': self.init_params,
            'alg': alg_name,
            'lr': self.optimizer_lr
        })

        joblib.dump(dump_dict, os.path.join(results_dir, f'{alg_name}_{seed}'))