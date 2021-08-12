import os
import argparse
import joblib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core, Logger
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.dataset import compute_J

from custom_env.ball_rolling_gym_env import BallRollingGym
from custom_policy.promp_policy import ProMPPolicy
from mushroom_rl.environments import Gym

from utils_el import init_distribution, init_algorithm, log_constraints

import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        super(Network, self).__init__()
        hidden_features = 32 # in_features[0] // 2
        layers = [nn.Linear(in_features[0], hidden_features)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_features, out_features=out_features[0])]
        layers += [nn.Tanh()]
        self.layers = nn.Sequential(*layers)

    def forward(self,x) -> torch.Tensor:
        return self.layers(x)

def experiment( env_name, horizon, env_gamma, \
                alg, eps, kappa, k, \
                sigma_init, distribution, \
                method, mi_type, bins, sample_type, gamma, mi_avg, \
                n_epochs, fit_per_epoch, ep_per_fit, \
                seed, results_dir, quiet):
                
    quiet = bool(quiet)
    mi_avg = bool(mi_avg)
    init_params = locals()
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # MDP
    mdp = Gym(env_name, horizon=None if horizon == 0 else horizon, gamma=0.99)
    
    approximator = Regressor(TorchApproximator,
                             network = Network,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)
                             
    policy = DeterministicPolicy(mu=approximator)

    # init distribution
    distribution = init_distribution(mu_init=0, sigma_init=sigma_init, size=policy.weights_size, sample_type=sample_type, gamma=gamma, distribution_class=distribution)
    print(f'{policy.weights_size} parameters')

    # init agent
    alg, params = init_algorithm(algorithm_class=alg, params=init_params)
    agent = alg(mdp.info, distribution, policy, features=None, **params)

    # train
    core = Core(agent, mdp)

    dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    R = compute_J(dataset_eval, gamma=1.)
    print('J at start : ' + str(np.mean(J)))
    
    returns_mean = [np.mean(J)]
    reward_mean = [np.mean(R)]
    returns_std = [np.std(J)]

    for i in range(n_epochs):
        
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit, quiet=quiet)

        dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
        # print('distribution parameters: ', distribution.get_parameters())
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        R = compute_J(dataset_eval, gamma=1.)
        print('J at iteration ' + str(i) + ': ' + str(round(np.mean(J),4)))
        
        returns_mean += [np.mean(J)]
        reward_mean += [np.mean(R)]
        returns_std += [np.std(J)]

        # logging inside
        gain_policy = policy.get_weights()
        mus, kls, entropys, mi_avg = log_constraints(agent)
        best_reward = np.array(returns_mean).max()

        dump_dict = dict({
            'returns_mean': returns_mean,
            'reward_mean': reward_mean,
            'returns_std': returns_std,
            'gain_policy': gain_policy,
            'best_reward': best_reward,
            'init_params': init_params,
            'alg': alg,
            'mi_avg': mi_avg,
            'mus': mus,
            'kls': kls,
            'entropys': entropys
        })

        joblib.dump(dump_dict, os.path.join(results_dir, f'{alg.__name__}_{seed}'))
        
        dump_state = dict({
            'distribution': distribution
        })

        joblib.dump(dump_state, os.path.join(results_dir, f'{alg.__name__}_{seed}_state'))

        filename = os.path.join(results_dir, f'log_{alg.__name__}_{seed}.txt')
        os.makedirs(results_dir, exist_ok=True)
        with open(filename, 'w') as file:
            for key in init_params.keys():
                file.write(f'{key}: {init_params[key]}\n')

    # logging
    gain_policy = policy.get_weights()
    mus, kls, entropys, mi_avg = log_constraints(agent)
    best_reward = np.array(returns_mean).max()

    dump_dict = dict({
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        'gain_policy': gain_policy,
        'best_reward': best_reward,
        'init_params': init_params,
        'alg': alg,
        'mi_avg': mi_avg,
        'mus': mus,
        'kls': kls,
        'entropys': entropys
    })

    joblib.dump(dump_dict, os.path.join(results_dir, f'{alg.__name__}_{seed}'))
    
    dump_state = dict({
        'distribution': distribution
    })

    joblib.dump(dump_state, os.path.join(results_dir, f'{alg.__name__}_{seed}_state'))

    filename = os.path.join(results_dir, f'log_{alg.__name__}_{seed}.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(filename, 'w') as file:
        for key in init_params.keys():
            file.write(f'{key}: {init_params[key]}\n')

def default_params():
    defaults = dict(
        # environment
        env_name = 'HopperBulletEnv-v0',
        horizon = 0, # None
        env_gamma = 0.99,

        # algorithm
        alg = 'ConstrainedREPSMIFull',
        eps = 2.7,
        kappa = 2.,
        k = 200, # 1/3 of parameters

        # distribution
        sigma_init = 3e-1,
        distribution = 'mi',

        # MI related
        method ='MI', # Pearson
        mi_type = 'regression',
        bins = 4,
        sample_type = None,
        gamma = 0.1,
        mi_avg = 0, # False

        # training
        n_epochs = 50,
        fit_per_epoch = 4, 
        ep_per_fit = 3000,

        # misc
        seed = 0,
        results_dir = 'results',
        quiet = 1 # True
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env-name', type=str)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--env-gamma', type=float)

    parser.add_argument('--alg', type=str)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--kappa', type=float)
    parser.add_argument('--k', type=int)

    parser.add_argument('--sigma-init', type=float)
    parser.add_argument('--distribution', type=str)

    parser.add_argument('--method', type=str)
    parser.add_argument('--mi-type', type=str)
    parser.add_argument('--bins', type=int)
    parser.add_argument('--sample-type', type=str)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--mi-avg', type=int)

    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--fit-per-epoch', type=int)
    parser.add_argument('--ep-per-fit', type=int)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--quiet', type=int)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
