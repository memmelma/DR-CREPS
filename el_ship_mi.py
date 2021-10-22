import os
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles

from mushroom_rl.environments import ShipSteering

from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor

from utils_el import init_distribution, init_algorithm, log_constraints

def experiment( n_tilings, \
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
    mdp = ShipSteering()

    # Policy
    high = [150, 150, np.pi]
    low = [0, 0, -np.pi]
    # low = [0, 0, -np.pi, -0.26]
    # high = [150, 150, np.pi, 0.26]
    n_tiles = [5, 5, 6]
    # n_tiles = [5, 5, 6, 5]
    low = np.array(low, dtype=float)
    high = np.array(high, dtype=float)
    n_tilings = n_tilings

    tilings = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low,
                             high=high)

    features = Features(tilings=tilings)
    input_shape = (features.size,)
    
    # Oracle
    try:
        approximator = Regressor(LinearApproximator, input_shape=input_shape,
                                output_shape=mdp.info.action_space.shape)
        policy = DeterministicPolicy(approximator)
        alg_string = alg
        distribution_string = distribution
        distribution = joblib.load(f'logs/ship/all_best_25/alg_ConstrainedREPSMI/k_75/sample_type_percentage/gamma_0.9/eps_5.3/kappa_14.0/ConstrainedREPSMI_0_state')['distribution']
        alg, params = init_algorithm(algorithm_class='ConstrainedREPSMI', params=init_params)
        agent = alg(mdp.info, distribution, policy, features=features, **params)
        core = Core(agent, mdp)

        alg = alg_string
        distribution = distribution_string
        dataset_eval = core.evaluate(n_episodes=1, quiet=quiet)
        
        oracle = np.arange(0, policy.weights_size, 1)[agent.states > 0].tolist()
        print('Successfully loaded Oracle!')
    except:
        if 'Oracle' in alg or 'ORACLE' in alg:
            print('Failed to load Oracle!')
            exit()
        oracle=None
    
    init_params['oracle'] = oracle

    # init approximator
    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape)
    policy = DeterministicPolicy(approximator)


    # init distribution
    distribution = init_distribution(mu_init=0, sigma_init=sigma_init, size=policy.weights_size, sample_type=sample_type, gamma=gamma, distribution_class=distribution)
    
    print('action space', mdp.info.action_space.shape)
    print('parameters', policy.weights_size)

    # TODO
    # distribution = joblib.load('logs/ship/all_best_25/alg_ConstrainedREPSMI/k_75/sample_type_percentage/gamma_0.9/eps_5.3/kappa_14.0/ConstrainedREPSMI_0_state')['distribution']
    # distribution = joblib.load('C:/Users/Marius/iprl_bbo/logs/ship/ConstrainedREPSMI_0_state')['distribution']
    
    # init agent
    alg, params = init_algorithm(algorithm_class=alg, params=init_params)
    agent = alg(mdp.info, distribution, policy, features=features, **params)

    # train
    core = Core(agent, mdp)

    dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet, render=not quiet)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    returns_mean = [np.mean(J)]
    returns_std = [np.std(J)]

    for i in range(n_epochs):
        
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit, quiet=quiet)

        dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet, render=not quiet)
        # print('distribution parameters: ', distribution.get_parameters())
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(round(np.mean(J),4)))
        
        returns_mean += [np.mean(J)]
        returns_std += [np.std(J)]

    agent.states = np.zeros(policy.weights_size)
    dataset_eval = core.evaluate(n_episodes=1, quiet=quiet, render=not quiet)
    print(agent.states.astype(np.int32))

    # logging
    gain_policy = policy.get_weights()
    mus, kls, entropys, mi_avg = log_constraints(agent)
    best_reward = np.array(returns_mean).max()

    dump_dict = dict({
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        # 'agent': agent,
        'gain_policy': gain_policy,
        'best_reward': best_reward,
        'optimal_reward': np.ones_like(best_reward)*-59,
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
        n_tilings = 1,

        # algorithm
        # alg = 'REPS_MI_ORACLE',
        # alg = 'ConstrainedREPSMIOracle',
        # alg = 'ConstrainedREPSMI',
        # alg = 'REPS_MI',
        alg = 'REPS',
        eps = 0.9,
        kappa = 14.0,
        k = 8,

        # distribution
        sigma_init = 7e-2,
        distribution = 'diag',

        # MI related
        method = 'MI', # Pearson
        mi_type = 'regression',
        bins = 4,
        sample_type = None,
        gamma = 0.9,
        mi_avg = 0, # False

        # training
        n_epochs = 10,
        fit_per_epoch = 1, 
        ep_per_fit = 25,

        # misc
        seed = 2,
        results_dir = 'results',
        quiet = 0 # True
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n-tilings', type=int)

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
