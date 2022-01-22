# NES code adapted from https://github.com/goktug97/nes-torch
# ES code adapted from https://github.com/alirezamika/evostra/blob/master/evostra/algorithms/evolution_strategy.py

import os
import sys
import torch
import numpy as np

from copy import deepcopy

from .utils import *

import joblib

class CoreNES():
    """
    """
    def __init__(self, policy, mdp,  alg='nes', optimizer=torch.optim.Adam, optimizer_lr=0.02, 
                    n_step=300, n_rollout=2, prepro=None, seed=42):


        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.alg = alg
        self.env = mdp
        self.horizon = mdp.info.horizon

        self.prepro = prepro

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
        gamma = self.env.info.gamma
        total_reward = 0
        for _ in range(self.n_rollout):
            done = False
            obs = self.env.reset()
            policy.set_weights(torch.nn.utils.parameters_to_vector(policy.parameters()))

            horizon_counter = 0
            while not done:
                if self.prepro is not None:
                    obs = self.prepro(obs)
                action = policy.draw_action(obs)
                obs, reward, done, _ = self.env.step(action.numpy())
                
                # if self.horizon is not None:
                horizon_counter += 1
                if horizon_counter == self.horizon:
                    done = True

                total_reward += gamma ** horizon_counter * reward

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

    def train(self):
        """Train ``self.policy`` for ``self.config.nes.n_steps`` to increase reward returns
        from the ``self.env`` using Natural Evolution Strategy gradient estimation."""
        torch.set_grad_enabled(False)
     
        n_samples = 0
        
        # initial evaluation
        reward = self.evaluate_policy(self.policy)
        self.rewards_mean += [reward]
        print(f'Gen: 0 Test Reward: {reward} Best Reward: {self.best_reward}')

        for gen in range(self.n_step):
            self.gen = gen

            # Sample
            population_params, epsilons = self.policy.sample()

            # Evaluate Population
            rewards = self.evaluate_populations(population_params)

            if self.alg == 'NES':
                # Calculate Gradients
                grad = self.policy.calculate_gradients(rewards, epsilons)
                self.optimize(grad)

            elif self.alg == 'ES':
                self.policy.no_grad_update(rewards, population_params, self.optim)
            else:
                return

            n_samples += self.policy.population_size*self.n_rollout
            print(f'samples total: {n_samples}')
            
            reward = self.evaluate_policy(self.policy)

            self.rewards_mean += [reward]
            
            if reward > self.best_reward:
                self.best_reward = reward
            print(f'Gen: {self.gen+1} Test Reward: {reward} Best Reward: {self.best_reward}')

        sys.stdout = sys.__stdout__
        torch.set_grad_enabled(True)
