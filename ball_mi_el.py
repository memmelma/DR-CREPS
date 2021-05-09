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
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.optimizers import AdaptiveOptimizer
from mushroom_rl.solvers.lqr import compute_lqr_feedback_gain

from mushroom_rl.features import Features
from mushroom_rl.features.basis.polynomial import PolynomialBasis
    
from custom_env.ball_rolling_gym_env import BallRollingGym
from custom_policy.promp_policy import ProMPPolicy

from mushroom_rl.algorithms.policy_search.black_box_optimization import REPS

from custom_algorithms.constrained_reps import ConstrainedREPS
from custom_algorithms.constrained_reps_mi import ConstrainedREPSMI

from custom_distributions.gaussian_custom import GaussianDiagonalDistribution, GaussianCholeskyDistribution

def experiment(alg, eps, k, kappa, gamma, n_epochs, fit_per_epoch, ep_per_fit, n_basis=20, horizon=1000, sigma_init=1e-3, seed=42, sample_type=None, results_dir='results', quiet=True):
    
    # MDP
    mdp = BallRollingGym(horizon=horizon, gamma=0.99, observation_ids=[0,1,2,3], render=False)

    init_params = locals()
    
    os.makedirs(results_dir, exist_ok=True)
    
    # basis_features = PolynomialBasis().generate(2, mdp.info.observation_space.shape[0])
    # features = Features(basis_features)
    features=None

    # approximator = Regressor(LinearApproximator,
    #                          input_shape=mdp.info.observation_space.shape,
    #                          output_shape=mdp.info.action_space.shape)

    # policy = DeterministicPolicy(mu=approximator)

    policy = ProMPPolicy(n_basis=n_basis, basis_width=0.001, maxSteps=horizon, output=mdp.info.action_space.shape)

    mu = np.zeros(policy.weights_size)
    # Cholesky
    # sigma = 1e-3 * np.eye(policy.weights_size)
    # distribution = GaussianCholeskyDistribution(mu, sigma)
    # Diag
    std = sigma_init * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, std)
    # Gaussian w/ fixed cov
    # sigma = 1e-3 * np.eye(policy.weights_size)
    # distribution = GaussianDistribution(mu, sigma)

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
        params = {'eps': eps, 'k': k}

    elif alg == 'REPS_MI_ORACLE':
        alg = REPS_MI
        oracle = []
        for i in range(lqr_dim):
            if i not in ineff_params:
                for j in range(lqr_dim):
                    oracle += [i*lqr_dim + j]
        print(oracle)
        params = {'eps': eps, 'k': k, 'oracle': oracle}

    # constrained
    elif alg == 'ConstrainedREPS':
        alg = ConstrainedREPS
        params = {'eps': eps, 'kappa': kappa}

    elif alg == 'ConstrainedREPSMI':
        alg = ConstrainedREPSMI

        params = {'eps': eps, 'k': k, 'kappa': kappa}

    elif alg == 'ConstrainedREPSMIOracle':
        alg = ConstrainedREPSMI
        oracle = []
        for i in range(lqr_dim):
            if i not in ineff_params:
                for j in range(lqr_dim):
                    oracle += [i*lqr_dim + j]
        print(oracle)
        params = {'eps': eps, 'k': k, 'kappa': kappa, 'oracle': oracle}

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

    mi_avg = None
    if 'MI' in alg.__name__:
        mi_avg = agent.mis

    best_reward = np.array(returns_mean).max()

    mus = agent.mus
    kls = agent.kls

    dump_dict = dict({
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        'agent': agent,
        'gain_policy': gain_policy,
        'best_reward': best_reward,
        'init_params': init_params,
        'alg': alg,
        'mi_avg': mi_avg,
        'mus': mus,
        'kls': kls,
        'ineff_params': ineff_params
    })

    joblib.dump(dump_dict, os.path.join(results_dir, f'{alg.__name__}_{seed}'))
    
    dump_state = dict({
        'mdp': mdp,
        'policy': policy,
        'distribution': distribution,
        'agent': agent,
        'core': core
    })

    joblib.dump(dump_state, os.path.join(results_dir, f'{alg.__name__}_{seed}_state'))

    filename = os.path.join(results_dir, f'log_{alg.__name__}_{seed}.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(filename, 'w') as file:
        for key in init_params.keys():
            file.write(f'{key}: {init_params[key]}\n')


def default_params():
    defaults = dict(
        alg = 'ConstrainedREPS',
        # alg = 'REPS_MI_CON_ORACLE',
        eps = 0.5,
        k = 1,
        kappa = 3,
        gamma= 0.1,
        n_epochs = 50, 
        fit_per_epoch = 1, 
        ep_per_fit = 10,
        n_basis=20,
        horizon=1000,
        sigma_init=1e-1,
        seed = 0,
        sample_type = None,
        results_dir = 'results',
        quiet = True
    )

    return defaults

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--alg', type=str)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--k', type=int)
    parser.add_argument('--kappa', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--fit-per-epoch', type=int)
    parser.add_argument('--ep-per-fit', type=int)
    parser.add_argument('--n-basis', type=int)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--sigma-init', type=float)
    parser.add_argument('--sample-type', type=str)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--quiet', type=bool)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
