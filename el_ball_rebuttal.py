import os
import argparse
import joblib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from custom_env.ball_rolling_gym_env import BallRollingGym
from custom_policy.promp_policy import ProMPPolicy

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor

from mushroom_rl.algorithms.actor_critic import PPO, TRPO
from mushroom_rl_benchmark.builders.actor_critic.deep_actor_critic import PPOBuilder, TRPOBuilder

from mushroom_rl.algorithms.policy_search import REINFORCE, GPOMDP, eNAC
from mushroom_rl.utils.optimizers import AdaptiveOptimizer
from mushroom_rl.policy import StateStdGaussianPolicy

from utils_el import init_distribution, init_algorithm, log_constraints

def experiment( n_basis, horizon, \
                alg, eps, kappa, k, \
                sigma_init, distribution, \
                method, mi_type, bins, sample_type, gamma, mi_avg, \
                n_epochs, fit_per_epoch, ep_per_fit, \
                seed, results_dir, quiet, save_render_path):
                
    quiet = bool(quiet)
    mi_avg = bool(mi_avg)
    init_params = locals()
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # MDP
    mdp = BallRollingGym(horizon=horizon, gamma=0.99, observation_ids=[0,1,2,3], render=not quiet, save_render_path=save_render_path)
    
    ## TRPO / PPO
    if alg == 'PPO':
        nn_policy = True
        alg = PPO
        agent_builder = PPOBuilder.default(
            # actor_lr=3e-4,
            # critic_lr=3e-4,
            actor_lr=eps, # 3e-3,
            critic_lr=eps, # 3e-3,
            n_features=32
        )
        agent = agent_builder.build(mdp.info)
    
    elif alg == 'TRPO':
        nn_policy = True
        alg = TRPO
        agent_builder = TRPOBuilder.default(
            # critic_lr=3e-3,
            # max_kl=1e-2,
            critic_lr= eps, # 3e-2,
            max_kl= kappa, # 1e-1,
            n_features=32
        )
        agent = agent_builder.build(mdp.info)

    ## Policy Gradient
    elif alg == 'REINFORCE':
        nn_policy = True
        approximator = Regressor(LinearApproximator,
                                input_shape=mdp.info.observation_space.shape,
                                output_shape=mdp.info.action_space.shape)

        sigma = Regressor(LinearApproximator,
                        input_shape=mdp.info.observation_space.shape,
                        output_shape=mdp.info.action_space.shape)

        # sigma_weights = sigma_init * np.ones(sigma.weights_size)
        # sigma.set_weights(sigma_weights)

        policy = StateStdGaussianPolicy(approximator, sigma)

        alg = REINFORCE # REINFORCE GPOMDP eNAC
        optimizer = AdaptiveOptimizer(eps=eps) # 1e-2
        algorithm_params = dict(optimizer=optimizer)
        agent = alg(mdp.info, policy, **algorithm_params)
    else:
        # init distribution
        policy = ProMPPolicy(n_basis=n_basis, basis_width=1e-3, maxSteps=horizon, output=mdp.info.action_space.shape)
        distribution = init_distribution(mu_init=0, sigma_init=sigma_init, size=policy.weights_size, sample_type=sample_type, gamma=gamma, distribution_class=distribution)
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
    mus, kls, entropys, mi_avg = log_constraints(agent)
    best_reward = np.array(returns_mean).max()

    dump_dict = dict({
        'returns_mean': returns_mean,
        'returns_std': returns_std,
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
        n_basis = 20,
        horizon = 750,

        # algorithm
        alg = 'REINFORCE',
        eps = 1e-3,
        kappa = 0,
        k = 0,

        # distribution
        sigma_init = 1.,
        distribution = None,

        # MI related
        method = None,
        mi_type = None,
        bins = 0,
        sample_type = None,
        gamma = 0,
        mi_avg = 0,
        
        n_epochs = 3,
        fit_per_epoch = 1, 
        ep_per_fit = 5,

        # misc
        seed = 0,
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
