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
from mushroom_rl.utils.optimizers import AdaptiveOptimizer
from mushroom_rl.solvers.lqr import compute_lqr_feedback_gain

from mushroom_rl.algorithms.policy_search.black_box_optimization import REPS, RWR
from custom_algorithms.more import MORE
from custom_algorithms.reps_mi import REPS_MI

from custom_algorithms.constrained_reps import ConstrainedREPS
from custom_algorithms.constrained_reps_mi import ConstrainedREPSMI

from custom_distributions.gaussian_custom import GaussianDiagonalDistribution, GaussianCholeskyDistribution

def experiment(alg, lqr_dim, eps, k, bins, kappa, gamma, n_epochs, fit_per_epoch, ep_per_fit, sigma_init=1e-3, env_seed=42, n_ineff=-1, seed=42, sample_type=None, mi_type='regression', mi_avg=1, results_dir='results', quiet=True):
    
    mi_avg = bool(mi_avg)

    if n_ineff < 0:
        n_ineff = round(lqr_dim / 2)
    np.random.seed(seed)
    
    horizon = 50
    mdp = LQR.generate(dimensions=lqr_dim, horizon=horizon, episodic=False, max_pos=1., max_action=1., eps=0.1)# eps=0.5)

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
    
    # Calculate optimal control return
    gain_lqr = compute_lqr_feedback_gain(mdp)
    state = mdp.reset()
    optimal_reward = 0
    for i in range(horizon):
        action = - gain_lqr @ state
        state, reward, _, __ = mdp.step(action)
        optimal_reward += reward
    
    # mdp.Q[mdp.Q == 1e-10] = 0
    # mdp.R[mdp.R == 1e-10] = 0
    # mdp.B[mdp.B == 1e-10] = 0

    init_params = locals()
    
    os.makedirs(results_dir, exist_ok=True)
    
    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)

    sigma = sigma_init * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    # sample type
    if sample_type == 'fixed':
            distribution.set_fixed_sample(gamma=gamma)
    elif sample_type == 'percentage':
        distribution.set_percentage_sample(gamma=gamma)
    elif sample_type == 'importance':
        distribution.set_importance_sample()

    # algorithms
    if alg == 'REPS':
        alg = REPS
        params = {'eps': eps}

    elif alg == 'REPS_MI':
        alg = REPS_MI
        params = {'eps': eps, 'gamma': gamma, 'k': k, 'bins': bins, 'mi_type': mi_type, 'mi_avg': mi_avg}

    elif alg == 'REPS_MI_ORACLE':
        alg = REPS_MI
        oracle = []
        for i in range(lqr_dim):
            if i not in ineff_params:
                for j in range(lqr_dim):
                    oracle += [i*lqr_dim + j]
        print(oracle)
        params = {'eps': eps, 'k': k, 'bins': bins, 'mi_type': mi_type, 'mi_avg': mi_avg, 'oracle': oracle}

    # constrained
    elif alg == 'ConstrainedREPS':
        alg = ConstrainedREPS
        params = {'eps': eps, 'kappa': kappa}

    elif alg == 'ConstrainedREPSMI':
        alg = ConstrainedREPSMI

        params = {'eps': eps, 'k': k, 'kappa': kappa, 'gamma': gamma, 'bins': bins, 'mi_type': mi_type, 'mi_avg': mi_avg}

    elif alg == 'ConstrainedREPSMIOracle':
        alg = ConstrainedREPSMI
        oracle = []
        for i in range(lqr_dim):
            if i not in ineff_params:
                for j in range(lqr_dim):
                    oracle += [i*lqr_dim + j]
        print(oracle)
        params = {'eps': eps, 'k': k, 'kappa': kappa, 'gamma': gamma, 'oracle': oracle, 'bins': bins, 'mi_type': mi_type}

    elif alg == 'MORE':
        alg = MORE
        params = {'eps': eps, 'kappa': kappa}
    
    elif alg == 'RWR':
        alg = RWR
        params = {'beta': eps}

    # Agent
    agent = alg(mdp.info, distribution, policy, **params)

    # Train
    core = Core(agent, mdp)

    dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))
    
    returns_mean = [np.mean(J)]
    returns_std = [np.std(J)]

    for i in range(n_epochs):
        
        # from mushroom_rl.utils.parameters import to_parameter
        # print(policy.weights_size)
        # if hasattr(core.agent, '_k'):
        #     if i+k > policy.weights_size:
        #         core.agent._k = to_parameter(policy.weights_size)
        #     else:
        #         core.agent._k = to_parameter(i+k)

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
    entropys = None
    mi_avg = None

    if hasattr(agent, 'mis'):
        mi_avg = agent.mis
    if hasattr(agent, 'mus'):
        mus = agent.mus
    if hasattr(agent, 'kls'):
        kls = agent.kls
    if hasattr(agent, 'entropys'):
        entropys = agent.entropys

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
        alg = 'REPS_MI',
        # alg = 'MORE',
        lqr_dim = 3,
        eps = 0.7,
        k = 2,
        bins = 4,
        kappa = 2,
        gamma = 0.1,
        n_epochs = 4,
        fit_per_epoch = 1, 
        ep_per_fit = 25,
        sigma_init=1e-1,
        seed = 0,
        n_ineff = 0,
        env_seed = -1,
        sample_type = 'importance',
        mi_type = 'regression',
        mi_avg = 0, # False
        results_dir = 'results',
        quiet = True
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--alg', type=str)
    parser.add_argument('--lqr-dim', type=int)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--k', type=int)
    parser.add_argument('--bins', type=int)
    parser.add_argument('--kappa', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--fit-per-epoch', type=int)
    parser.add_argument('--ep-per-fit', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--env-seed', type=int)
    parser.add_argument('--n-ineff', type=int)
    parser.add_argument('--sigma-init', type=float)
    parser.add_argument('--sample-type', type=str)
    parser.add_argument('--mi-type', type=str)
    parser.add_argument('--mi-avg', type=int)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--quiet', type=bool)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
