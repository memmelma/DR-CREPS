import os
import argparse
import joblib
from tqdm import tqdm
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.solvers.lqr import compute_lqr_feedback_gain

from custom_algorithms.reps_mi import REPS_MI
from custom_algorithms.constrained_reps_mi import ConstrainedREPSMI

from utils_el import init_distribution, init_algorithm, log_constraints

def experiment( lqr_dim, n_ineff, env_seed, \
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

    # init LQR
    horizon = 50
    mdp = LQR.generate(dimensions=lqr_dim, horizon=50, episodic=False, max_pos=1., max_action=1., eps=0.1)# eps=0.5)

    # init reduced LQR
    if env_seed >= 0:
        rng = default_rng(seed=env_seed)
        ineff_params = rng.choice(lqr_dim, size=n_ineff, replace=False)
        for p in ineff_params:
            # mdp.A[p][p] = 1e-10
            mdp.B[p][p] = 1e-20
            mdp.Q[p][p] = 1e-20
            # mdp.R[p][p] = 1e-20
        print('\nA', mdp.A, '\nB', mdp.B, '\nQ', mdp.Q, '\nR', mdp.R, '\nineff_params', ineff_params)
    else:
        ineff_params = []
    
    # compute optimal control return
    gain_lqr = compute_lqr_feedback_gain(mdp)
    state = mdp.reset()
    optimal_reward = 0
    for i in range(horizon):
        action = - gain_lqr @ state
        state, reward, _, __ = mdp.step(action)
        optimal_reward += reward
    
    # mdp.Q[mdp.Q == 1e-20] = 0
    # mdp.R[mdp.R == 1e-20] = 0
    # mdp.B[mdp.B == 1e-20] = 0
    
    # init lower level policy
    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)
    policy = DeterministicPolicy(mu=approximator)

    # init distribution
    distribution = init_distribution(mu_init=0, sigma_init=sigma_init, size=policy.weights_size, sample_type=sample_type, gamma=gamma, distribution_class=distribution)

    # init algorithm
    if alg == 'REPS_MI_ORACLE':
        alg = REPS_MI
        oracle = []
        for i in range(lqr_dim):
            if i not in ineff_params:
                for j in range(lqr_dim):
                    oracle += [i*lqr_dim + j]
        print(oracle)
        params = {'eps': eps, 'k': k, 'bins': bins, 'mi_type': mi_type, 'mi_avg': mi_avg, 'oracle': oracle}

    elif alg == 'ConstrainedREPSMIOracle':
        alg = ConstrainedREPSMI
        oracle = []
        for i in range(lqr_dim):
            if i not in ineff_params:
                for j in range(lqr_dim):
                    oracle += [i*lqr_dim + j]
        print(oracle)
        params = {'eps': eps, 'k': k, 'kappa': kappa, 'gamma': gamma, 'oracle': oracle, 'bins': bins, 'mi_type': mi_type}


    # init agent
    else:
        alg, params = init_algorithm(algorithm_class=alg, params=init_params)
    agent = alg(mdp.info, distribution, policy, **params)

    # train
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
        'agent': agent,
        'gain_lqr': gain_lqr,
        'gain_policy': gain_policy,
        'optimal_reward': optimal_reward,
        'best_reward': best_reward,
        'init_params': init_params,
        'alg': alg,
        'mi_avg': mi_avg,
        'mus': mus,
        'kls': kls,
        'entropys': entropys,
        'ineff_params': ineff_params
    })

    joblib.dump(dump_dict, os.path.join(results_dir, f'{alg.__name__}_{seed}'))
    
    filename = os.path.join(results_dir, f'log_{alg.__name__}_{seed}.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(filename, 'w') as file:
        for key in init_params.keys():
            file.write(f'{key}: {init_params[key]}\n')


def default_params():
    defaults = dict(
        # environment
        lqr_dim = 10,
        n_ineff = 3,
        env_seed = -1,

        # algorithm
        alg = 'ConstrainedREPSMIFull',
        eps = 2.2,
        kappa = 2,
        k = 5,

        # distribution
        sigma_init = 1e-1,
        distribution = 'diag',

        # MI related
        method = 'MI', # Pearson
        mi_type = 'regression',
        bins = 4,
        sample_type = None,
        gamma = 0.1,
        mi_avg = 0, # False

        # training
        n_epochs = 4,
        fit_per_epoch = 1, 
        ep_per_fit = 25,

        # misc
        seed = 0,
        results_dir = 'results',
        quiet = 1 # True
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lqr-dim', type=int)
    parser.add_argument('--n-ineff', type=int)
    parser.add_argument('--env-seed', type=int)

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
