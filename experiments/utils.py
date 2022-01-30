import os
import joblib
import numpy as np

from algorithms import DR_CREPS_PE, DR_REPS_PE, RWR_PE, CEM, MORE
from distributions import GaussianDiagonalDistribution, GaussianDistributionGDR, GaussianCholeskyDistribution
from mushroom_rl.distributions import GaussianDistribution

from mushroom_rl.distributions.distribution import Distribution
from mushroom_rl.algorithms.policy_search.black_box_optimization import REPS, RWR, ConstrainedREPS

from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor

from mushroom_rl.algorithms.actor_critic import PPO, TRPO
from mushroom_rl_benchmark.builders.actor_critic.deep_actor_critic import PPOBuilder, TRPOBuilder

from mushroom_rl.algorithms.policy_search import REINFORCE
from mushroom_rl.utils.optimizers import AdaptiveOptimizer
from mushroom_rl.policy import StateStdGaussianPolicy

def init_distribution(mu_init=0, sigma_init=1e-3, size=1, sample_strat=None, lambd=0.0, distribution_class='diag'):
    
    print('distribution_class', distribution_class)
    
    mu = mu_init * np.ones(size)

    if type(sigma_init) != float and type(sigma_init) is Distribution:
        distribution = sigma_init
        print(f'Successfully loaded distribution {type(sigma_init)}!')

    elif distribution_class == 'diag':
        sigma = sigma_init * np.ones(size)
        distribution = GaussianDiagonalDistribution(mu, sigma)
    else:
        sigma = sigma_init**2 * np.eye(size)
        if distribution_class == 'cholesky':
            distribution = GaussianCholeskyDistribution(mu, sigma)
        elif distribution_class == 'fixed':
            distribution = GaussianDistribution(mu, sigma)
        elif distribution_class == 'gdr':
            distribution = GaussianDistributionGDR(mu, sigma)

    if sample_strat is not None:
        assert distribution_class == 'diag' or distribution_class == 'gdr', \
            f"Argument 'sample_strat' only supported for distribution_class = 'diag' or 'gdr', got {distribution_class}!"
        distribution.set_sample_strat(sample_strat=sample_strat, lambd=lambd)

    return distribution

def init_policy_search_algorithm(algorithm_class='REPS', params={}):

    print('algorithm_class', algorithm_class)
    
    if algorithm_class == 'CEM':
        alg = CEM
        params = {'eps': params['eps']}

    elif algorithm_class == 'REPS':
        alg = REPS
        params = {'eps': params['eps']}

    elif algorithm_class == 'REPS-PE':
        alg = DR_REPS_PE
        params = {'eps': params['eps'], 'lambd': params['lambd'], 'k': params['k'],
                    'C': params['C'],'mi_estimator': params['mi_estimator'], 'gdr': False}

    elif algorithm_class == 'DR-REPS-PE':
        alg = DR_REPS_PE
        params = {'eps': params['eps'], 'lambd': params['lambd'], 'k': params['k'],
                    'C': params['C'], 'mi_estimator': params['mi_estimator'], 'gdr': True}
      
    elif algorithm_class == 'RWR':
        alg = RWR
        params = {'beta': params['eps']}
    
    elif algorithm_class == 'PRO':
        alg = RWR_PE
        params = {'eps': params['eps'], 'lambd': 0, 'k': 0,
                    'C': 'PCC', 'mi_estimator': None}

    elif algorithm_class == 'RWR-PE':
        alg = RWR_PE
        params = {'eps': params['eps'], 'lambd': params['lambd'], 'k': params['k'],
                    'C': params['C'], 'mi_estimator': params['mi_estimator']}

    elif algorithm_class == 'MORE':
        alg = MORE
        params = {'eps': params['eps'], 'kappa': params['kappa']}
        
    elif algorithm_class == 'CREPS':
        alg = ConstrainedREPS
        params = {'eps': params['eps'], 'kappa': params['kappa']}

    elif algorithm_class == 'CREPS-PE':
        alg = DR_CREPS_PE
        params = {'eps': params['eps'], 'k': params['k'], 'kappa': params['kappa'], 'lambd': params['lambd'],
                    'C': params['C'], 'mi_estimator': params['mi_estimator'], 'gdr': False}

    elif algorithm_class == 'DR-CREPS-PE':
        alg = DR_CREPS_PE
        params = {'eps': params['eps'], 'k': params['k'], 'kappa': params['kappa'], 'lambd': params['lambd'],
                    'C': params['C'], 'mi_estimator': params['mi_estimator'], 'gdr': True}
    
    else:
        raise Exception("Invalid algorithm selection. Select one of ['REPS', 'REPS-PE', 'DR-REPS-PE', 'RWR', 'PRO', 'RWR-PE', 'MORE', 'CREPS', 'CREPS-PE', 'DR-CREPS-PE'")

    return alg, params

def init_grad_agent(mdp, alg, actor_lr, critic_lr, max_kl, optim_eps, nn_policy=False):
    if alg == 'PPO':
        agent_builder = PPOBuilder.default(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            n_features=32
        )
        return agent_builder.build(mdp.info), PPO
    
    elif alg == 'TRPO':
        alg = TRPO
        agent_builder = TRPOBuilder.default(
            critic_lr= critic_lr,
            max_kl= max_kl,
            n_features=32
        )
        return agent_builder.build(mdp.info), TRPO

    ## Policy Gradient
    elif alg == 'REINFORCE':
        approximator = Regressor(LinearApproximator,
                                input_shape=mdp.info.observation_space.shape,
                                output_shape=mdp.info.action_space.shape)

        sigma = Regressor(LinearApproximator,
                        input_shape=mdp.info.observation_space.shape,
                        output_shape=mdp.info.action_space.shape)

        # sigma_weights = sigma_init * np.eye(sigma.weights_size)
        # sigma.set_weights(sigma_weights)

        policy = StateStdGaussianPolicy(approximator, sigma)

        return REINFORCE(mdp.info, policy, optimizer=AdaptiveOptimizer(eps=optim_eps)), REINFORCE

def save_results(dump_dict, results_dir, alg, init_params, seed):
    alg_name = alg.__name__ if type(alg) is not str else alg
    filename = os.path.join(results_dir, f'{alg_name}_{seed}')
    joblib.dump(dump_dict, filename)
    print(f'Results dumped at {os.path.join(os.getcwd(),filename)}')

    filename = os.path.join(results_dir, f'log_{alg_name}_{seed}.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(filename, 'w') as file:
        for key in init_params.keys():
            file.write(f'{key}: {init_params[key]}\n')
    print(f'Params dumped at {os.path.join(os.getcwd(),filename)}')
    