import os
import argparse
import joblib
import numpy as np
from numpy.random import default_rng

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
    mdp = LQR.generate(dimensions=lqr_dim, horizon=50, episodic=False, max_pos=1., max_action=1., eps=0.1)

    # init reduced LQR
    if env_seed >= 0:

        rng = default_rng(seed=env_seed)
        choice = np.arange(0,lqr_dim,1)
        rng.shuffle(choice)

        ineff_params = choice[:n_ineff]
        eff_params = choice[n_ineff:]

        oracle = []
        tmp = np.arange(0,lqr_dim**2,1)
        for p in eff_params:
            oracle += tmp[p*lqr_dim:(p+1)*lqr_dim].tolist()
        init_params['oracle'] = oracle

        for p in ineff_params:
            mdp.B[p][p] = 1e-20
            mdp.Q[p][p] = 1e-20
        
        if env_seed == 42:
            n_corr = 5 * lqr_dim
            corr_params_0 = rng.choice(eff_params, size=n_corr, replace=True)
            corr_params_1 = rng.choice(eff_params, size=n_corr, replace=True)
            for p_0, p_1 in zip(corr_params_0, corr_params_1):
                A = np.random.uniform(0, 0.5)
                B = np.random.uniform(0, 0.5)
                mdp.A[p_0][p_1] = A
                mdp.A[p_1][p_0] = A
                mdp.B[p_0][p_1] = B
                mdp.B[p_1][p_0] = B

    # env_seed < 0 for standard behavior
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
    
    # init lower level policy
    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)
    policy = DeterministicPolicy(mu=approximator)
    print('action space', mdp.info.action_space.shape)
    print('parameters', policy.weights_size)

    # init distribution
    distribution = init_distribution(mu_init=0, sigma_init=sigma_init, size=policy.weights_size, sample_type=sample_type, gamma=gamma, distribution_class=distribution)

    alg, params = init_algorithm(algorithm_class=alg, params=init_params)

    agent = alg(mdp.info, distribution, policy, **params)

    # train
    core = Core(agent, mdp)

    dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
    
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))
    
    returns_mean = [np.mean(J)]
    returns_std = [np.std(J)]

    for i in range(n_epochs):
        
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                n_episodes_per_fit=ep_per_fit, quiet=quiet)

        dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
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
        n_ineff = 7,
        env_seed = 1,

        # algorithm
        # alg = 'MORE',
        # alg = 'REPS_MI',
        # alg = 'REPS',
        # alg = 'REPS_MI_full',
        # alg = 'REPS_MI',
        alg = 'REPS_MI_ORACLE',
        # alg = 'ConstrainedREPSMIOracle',
        eps = .5,
        kappa = .5,
        k = 20,

        # distribution
        sigma_init = 3e-1,
        # distribution = 'diag',
        distribution = 'diag',
        # distribution = 'mi',

        # MI related
        method = 'MI', # Pearson
        mi_type = 'regression',
        bins = 4,
        sample_type = 'percentage',
        # sample_type = None,
        gamma = -1,#0.9,
        mi_avg = 0, # False

        # training
        n_epochs = 100,
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
