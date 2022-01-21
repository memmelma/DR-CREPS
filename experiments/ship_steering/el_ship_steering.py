import os
import joblib
import numpy as np

from mushroom_rl.environments import ShipSteering

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles

from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor

from experiments.utils import init_distribution, init_algorithm

def experiment_ship_steering(
    env, seed, env_seed, \
    lqr_dim, red_dim, \
    n_tilings, \
    n_basis, horizon, \
    alg, eps, kappa, k, \
    distribution, sigma_init, \
    C, mi_estimator, \
    sample_strat, lambd, \
    n_epochs, fit_per_epoch, ep_per_fit, \
    results_dir, save_render_path, verbose
):

    # misc
    verbose = bool(verbose==0)
    init_params = locals()
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # MDP
    mdp = ShipSteering()

    # features
    high = [150, 150, np.pi]
    low = [0, 0, -np.pi]
    # low = [0, 0, -np.pi, -0.26]
    # high = [150, 150, np.pi, 0.26]
    n_tiles = [5, 5, 6]
    # n_tiles = [5, 5, 6, 5]
    low = np.array(low, dtype=float)
    high = np.array(high, dtype=float)

    tilings = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, 
                            low=low, high=high)

    features = Features(tilings=tilings)
    input_shape = (features.size,)

    # parametric policy
    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape)
    policy = DeterministicPolicy(approximator)

    # search distribution
    distribution = init_distribution(mu_init=0, sigma_init=sigma_init, size=policy.weights_size, 
                                        sample_strat=sample_strat, lambd=lambd, distribution_class=distribution)
    
    # policy search algorithm
    alg, params = init_algorithm(algorithm_class=alg, params=init_params)
    agent = alg(mdp.info, distribution, policy, features=features, **params)

    # train
    core = Core(agent, mdp)

    dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=verbose, render=not verbose)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    returns_mean = [np.mean(J)]
    returns_std = [np.std(J)]

    for i in range(n_epochs):
        
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit, quiet=verbose)

        dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=verbose, render=not verbose)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(round(np.mean(J),4)))
        
        returns_mean += [np.mean(J)]
        returns_std += [np.std(J)]

    # logging
    best_reward = np.array(returns_mean).max()
    dump_dict = dict({
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        # 'agent': agent,
        'gain_policy': policy.get_weights(),
        'best_reward': best_reward,
        'optimal_reward': np.ones_like(best_reward)*-59,
        'init_params': init_params,
        'alg': alg
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
