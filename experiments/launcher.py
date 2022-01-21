import os
import argparse
import joblib
import numpy as np

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.solvers.lqr import compute_lqr_feedback_gain

from utils_el import init_distribution, init_algorithm

def experiment( lqr_dim, n_ineff, env_seed, \
                alg, eps, kappa, k, \
                sigma_init, distribution, \
                C, mi_estimator, sample_strat, lambd, \
                n_epochs, fit_per_epoch, ep_per_fit, \
                seed, results_dir, quiet):
    
    quiet = bool(quiet)
    init_params = locals()
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # init LQR
    horizon = 50
    mdp = LQR.generate(dimensions=lqr_dim, horizon=horizon, max_pos=1., max_action=1.)

    rng = np.random.default_rng(seed=0)
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
    
    # compute optimal control return
    gain_lqr = compute_lqr_feedback_gain(mdp, max_iterations=100000)
    state = mdp.reset()
    optimal_reward = 0
    for i in range(horizon):
        action = - gain_lqr @ state
        state, reward, _, __ = mdp.step(action)
        optimal_reward += reward
    print('optimal control', optimal_reward)
    
    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)
    policy = DeterministicPolicy(mu=approximator)

    print('action space', mdp.info.action_space.shape)
    print('action lo/hi', mdp.info.action_space.low, mdp.info.action_space.high)
    print('observation space', mdp.info.observation_space.shape)
    print('parameters', policy.weights_size)

    # init distribution
    distribution = init_distribution(mu_init=0, sigma_init=sigma_init, size=policy.weights_size, 
                                        sample_strat=sample_strat, lambd=lambd, distribution_class=distribution)

    alg, params = init_algorithm(algorithm_class=alg, params=init_params)

    agent = alg(mdp.info, distribution, policy, **params)

    # train
    core = Core(agent, mdp)

    dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
    
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))
    
    Js_mean = [np.mean(J)]
    Js_std = [np.std(J)]

    for i in range(n_epochs):
        
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                n_episodes_per_fit=ep_per_fit, quiet=quiet)

        dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(round(np.mean(J),4)))
        
        Js_mean += [np.mean(J)]
        Js_std += [np.std(J)]


    dump_dict = dict({
        'top_k_corr': agent.top_k_corr if hasattr(agent, 'top_k_corr') else None,
        'Js_mean': Js_mean,
        'Js_std': Js_std,
        'agent': agent,
        'gain_lqr': gain_lqr,
        'gain_policy': policy.get_weights(),
        'optimal_reward': optimal_reward,
        'best_reward': np.array(Js_mean).max(),
        'init_params': init_params,
        'alg': alg,
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
        env_seed = 0,

        # algorithm
        alg = 'PRO',
        eps = 1.,
        kappa = 3.5,
        k = 30,

        # distribution
        sigma_init = 3e-1,
        distribution = 'diag',

        # correlation measure
        C = 'PCC',
        mi_estimator = 'regression',
        sample_strat = 'PRO',
        lambd = 0.1,

        # training
        n_epochs = 10,
        fit_per_epoch = 1,
        ep_per_fit = 50,

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

    parser.add_argument('--C', type=str)
    parser.add_argument('--mi-estimator', type=str)
    parser.add_argument('--sample-strat', type=str)
    parser.add_argument('--lambd', type=float)

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
