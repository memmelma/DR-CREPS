import os
import argparse
import joblib
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles

from mushroom_rl.environments import ShipSteering

from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor

from mushroom_rl.algorithms.policy_search.black_box_optimization import REPS, RWR

from custom_algorithms.constrained_reps import ConstrainedREPS
from custom_algorithms.constrained_reps_mi import ConstrainedREPSMI

from custom_distributions.gaussian_custom import GaussianDiagonalDistribution

from custom_algorithms.more import MORE
from custom_algorithms.reps_mi import REPS_MI

def experiment(alg, eps, k, bins, kappa, gamma, n_epochs, fit_per_epoch, ep_per_fit, n_tilings=1, sigma_init=1e-3, seed=42, sample_type=None, mi_type='regression', mi_avg=True, results_dir='results', quiet=True):
    
    # MDP
    mdp = ShipSteering()

    init_params = locals()
    
    os.makedirs(results_dir, exist_ok=True)
      # Policy
    high = [150, 150, np.pi]
    low = [0, 0, -np.pi]
    n_tiles = [5, 5, 6]
    # n_tiles = [10, 10, 11]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = n_tilings

    tilings = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low,
                             high=high)

    features = Features(tilings=tilings)
    input_shape = (features.size,)

    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(approximator)

    # sigma_init = 4e-1
    print(policy.weights_size)
    mu = np.zeros(policy.weights_size)
    if type(sigma_init) == float:
        sigma = sigma_init * np.ones(policy.weights_size)
        distribution = GaussianDiagonalDistribution(mu, sigma)
    else:
        distribution = sigma_init

    print(distribution.get_parameters())
    
    # sample type
    if sample_type == 'fixed':
            distribution.set_fixed_sample(gamma=gamma)
    elif sample_type == 'percentage':
        distribution.set_percentage_sample(gamma=gamma)

    # algorithms
    if alg == 'REPS':
        alg = REPS
        params = {'eps': eps}
    
    elif alg == 'REPS_MI':
        alg = REPS_MI
        params = {'eps': eps, 'k': k, 'bins': bins, 'mi_type': mi_type, 'mi_avg': mi_avg}

    # constrained
    elif alg == 'ConstrainedREPS':
        alg = ConstrainedREPS
        params = {'eps': eps, 'kappa': kappa}

    elif alg == 'ConstrainedREPSMI':
        alg = ConstrainedREPSMI
        params = {'eps': eps, 'k': k, 'kappa': kappa, 'bins': bins, 'mi_type': mi_type, 'mi_avg': mi_avg}

    elif alg == 'MORE':
        alg = MORE
        params = {'eps': eps}
    
    elif alg == 'RWR':
        alg = RWR
        params = {'beta': eps}

    # Agent
    agent = alg(mdp.info, distribution, policy, features=features, **params)

    # Train
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
    

    gain_policy = policy.get_weights()

    mus = None
    kls = None
    mi_avg = None

    if hasattr(agent, 'mis'):
        mi_avg = agent.mis
    if hasattr(agent, 'mus'):
        mus = agent.mus
    if hasattr(agent, 'kls'):
        kls = agent.kls

    best_reward = np.array(returns_mean).max()

    del init_params['mdp']

    dump_dict = dict({
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        # 'agent': agent,
        'gain_policy': gain_policy,
        'best_reward': best_reward,
        'init_params': init_params,
        'alg': alg,
        'mi_avg': mi_avg,
        'mus': mus,
        'kls': kls
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
        alg = 'REPS_MI',
        eps = 1.,
        k = 75,
        bins = 3,
        kappa = 2,
        gamma= 0.1,
        n_epochs = 25, 
        fit_per_epoch = 1, 
        ep_per_fit = 25,
        n_tilings = 1,
        sigma_init = 4e-1,
        seed = 0,
        sample_type = None,
        mi_type = 'regression',
        mi_avg = True,
        results_dir = 'results',
        quiet = True
    )

    return defaults

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--alg', type=str)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--k', type=float)
    parser.add_argument('--bins', type=int)
    parser.add_argument('--kappa', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--fit-per-epoch', type=int)
    parser.add_argument('--ep-per-fit', type=int)
    parser.add_argument('--n-tilings', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--sigma-init', type=float)
    parser.add_argument('--sample-type', type=str)
    parser.add_argument('--mi-type', type=str)
    parser.add_argument('--mi-avg', type=str)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--quiet', type=bool)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)