import numpy as np
from tqdm import tqdm, trange
tqdm.monitor_interval = 0

from mushroom_rl.core import Core, Logger
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl.algorithms.policy_search import MORE
from mushroom_rl.algorithms.policy_search import REPS

from mushroom_rl.distributions.gaussian import GaussianDiagonalDistribution, GaussianCholeskyDistribution

from policy.pass_policy import PassPolicy
from environments.function import Function


def experiment(alg, params, distribution, n_epochs, fit_per_epoch, ep_per_fit, function_name, dim):
    np.random.seed(0)

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)
    
    mdp = Function(function_name, dim=dim)
    policy = PassPolicy(mu_shape=mdp.info.action_space.shape)

    mu = np.zeros(policy.weights_size)

    if distribution == GaussianDiagonalDistribution:
        sigma = np.sqrt(1e-1) * np.ones(policy.weights_size)
    else:
        sigma = 1e-1 * np.eye(policy.weights_size)

    distribution = distribution(mu, sigma)

    agent = alg(mdp.info, distribution, policy, **params)

    # train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_fit)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    logger.epoch_info(0, J=np.mean(J))

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit)
        dataset_eval = core.evaluate(n_episodes=ep_per_fit)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        logger.epoch_info(i+1, J=np.mean(J))


if __name__ == '__main__':
    
    function_names = ['styblinski', 'himmelblau', 'rosenbrock', 'rastrigin']

    algs = [MORE, REPS]
    params = [{'eps':.05, 'kappa':.05}, {'eps':0.05}]
    distributions = [GaussianCholeskyDistribution, GaussianCholeskyDistribution]

    for alg, params, distribution in zip(algs, params, distributions):
        experiment(alg, params, distribution, n_epochs=200, fit_per_epoch=1, ep_per_fit=100, function_name=function_names[0], dim=15)
