import os
import argparse
import joblib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl.environments.pybullet_envs.air_hockey import AirHockeyHit
from environments.ball_rolling_gym_env import BallRollingGym
from policy.promp_policy import ProMPPolicy

from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.policy import DeterministicPolicy

from utils_el import init_distribution, init_algorithm, log_constraints

from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

import torch
import torch.nn as nn
from mushroom_rl.utils.torch import get_weights

class Network(nn.Module):
    def __init__(self, input_shape, output_shape, second_hidden=False, *args, **kwargs) -> None:
        super(Network, self).__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        n_features = 16 # 8
        print(n_input, n_output)

        self.second_hidden = second_hidden

        self._h1 = nn.Linear(n_input, n_features)
        if self.second_hidden:
            self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        # nn.init.xavier_uniform_(self._h1.weight,
        #                         gain=nn.init.calculate_gain('relu'))
        # if self.second_hidden:
        #     nn.init.xavier_uniform_(self._h2.weight,
        #                             gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self._h3.weight,
        #                         gain=nn.init.calculate_gain('linear'))

    def forward(self, state) -> torch.Tensor:
        features1 = torch.relu(self._h1(torch.squeeze(state, 1).float()))
        
        if self.second_hidden:
            features2 = torch.relu(self._h2(features1))
        else:
            features2 = features1
        
        features3 = self._h3(features2)

        a = torch.tanh(features3)
        return a


def experiment( n_basis, horizon, \
                alg, eps, kappa, k, \
                sigma_init, distribution, \
                method, mi_type, bins, sample_type, gamma, mi_avg, \
                nn, \
                n_epochs, fit_per_epoch, ep_per_fit, \
                seed, results_dir, quiet, save_render_path):
                
    quiet = bool(quiet)
    mi_avg = bool(mi_avg)
    init_params = locals()
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # MDP
    # mdp = BallRollingGym(horizon=horizon, gamma=0.99, observation_ids=[0,1,2,3], render=not quiet)
    mdp = AirHockeyHit(horizon=horizon, debug_gui=not quiet, table_boundary_terminate=True)

    if save_render_path is not None:
        mdp.activate_save_render(save_render_path)


    if nn:
        approximator = Regressor(TorchApproximator,
                                network = Network,
                                input_shape=mdp.info.observation_space.shape,
                                output_shape=mdp.info.action_space.shape)
        policy = DeterministicPolicy(mu=approximator)
    else:
        policy = ProMPPolicy(n_basis=n_basis, basis_width=1e-3, maxSteps=horizon, output=mdp.info.action_space.shape)

    print('\nactions lo/hi', mdp.info.action_space.low, mdp.info.action_space.high)
    # init distribution
    distribution = init_distribution(mu_init=0, sigma_init=sigma_init, size=policy.weights_size, sample_type=sample_type, gamma=gamma, distribution_class=distribution)
    print('parameters', policy.weights_size)

    # if nn:
    #     distribution._mu = get_weights(approximator.model.network.parameters())

    # init agent
    alg, params = init_algorithm(algorithm_class=alg, params=init_params)
    agent = alg(mdp.info, distribution, policy, features=None, **params)
    
    if nn:
        prepro = StandardizationPreprocessor(mdp_info=mdp.info)

    # train
    if nn:
        core = Core(agent, mdp, preprocessors=[prepro])
    else:
        core = Core(agent, mdp)

    dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))
    
    returns_mean = [np.mean(J)]
    returns_std = [np.std(J)]

    for i in range(n_epochs):
        
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit, quiet=quiet)

        dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
        # print('distribution parameters: ', distribution.get_parameters())
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(round(np.mean(J),4)))
        
        returns_mean += [np.mean(J)]
        returns_std += [np.std(J)]
    
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
        'optimal_reward': np.ones_like(best_reward)*150.,
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
        n_basis = 30,
        horizon = 750,

        # algorithm
        # alg = 'ConstrainedREPS',
        alg = 'ConstrainedREPSMIFull',
        eps = 4.,
        kappa = 7.,
        k = 30,

        # distribution
        # sigma_init = 30.,
        # sigma_init = 1e-3,
        sigma_init = 3e-1,
        distribution = 'mi',

        # MI related
        method ='Pearson',
        mi_type = 'regression',
        bins = 4,
        sample_type = 'percentage',
        gamma = 0.1,
        mi_avg = 0, # False

        nn = 0, # False

        # training
        n_epochs = 200,
        fit_per_epoch = 1, 
        ep_per_fit = 50,

        # misc
        seed = 17, # MI 13 # 16
        results_dir = 'results',
        quiet = 1, # True
        save_render_path = None
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n-basis', type=int)
    parser.add_argument('--horizon', type=int)

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

    parser.add_argument('--nn', type=int)

    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--fit-per-epoch', type=int)
    parser.add_argument('--ep-per-fit', type=int)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--quiet', type=int)
    parser.add_argument('--save-render-path', type=int)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
