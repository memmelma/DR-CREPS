import numpy as np
from tqdm import tqdm

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core
from mushroom_rl.distributions import GaussianCholeskyDistribution, GaussianDiagonalDistribution, GaussianDistribution
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.optimizers import AdaptiveOptimizer
from mushroom_rl.algorithms.policy_search.black_box_optimization.reps import REPS
from mushroom_rl.solvers.lqr import compute_lqr_feedback_gain

from constrained_REPS import constrained_REPS
from more import MORE
from reps_mi import REPS_MI

import matplotlib.pyplot as plt

"""
This script aims to replicate the experiments on the LQR MDP using episode-based
policy search algorithms, also known as Black Box policy search algorithms.

"""

tqdm.monitor_interval = 0


def experiment(alg, mdp, params, n_epochs, fit_per_epoch, ep_per_fit, quiet=True):

    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)
    # sigma = 1e-3 * np.ones(policy.weights_size)
    # distribution = GaussianDiagonalDistribution(mu, sigma)

    sigma = 1e-3 * np.eye(policy.weights_size)
    distribution = GaussianCholeskyDistribution(mu, sigma)

    # Agent
    agent = alg(mdp.info, distribution, policy, **params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    # print('J at start : ' + str(np.mean(J)))
    
    returns_mean = [np.mean(J)]
    returns_std = [np.std(J)]

    for i in range(n_epochs):
        
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit, quiet=quiet)

        dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
        # print('distribution parameters: ', distribution.get_parameters())
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        # print('J at iteration ' + str(i) + ': ' + str(round(np.mean(J),4)))
        
        returns_mean += [np.mean(J)]
        returns_std += [np.std(J)]
    
    returns_mean = np.array(returns_mean)
    returns_std = np.array(returns_std)
    
    gain = compute_lqr_feedback_gain(mdp)
    print(policy.get_weights())
    print(gain)

    return returns_mean, returns_std, agent

def optimal_control(mdp):
    K = compute_lqr_feedback_gain(mdp, max_iterations=100)
    state = mdp.reset()
    action = K @ state
    state, reward, _, __ = mdp.step(action)
    print('optimal control reward for LQR env:', reward)

    # for i in range(25):
    #     state, reward, _, __ = mdp.step(action)
    #     print(reward)
    #     action = K @ state

if __name__ == '__main__':

    np.random.seed()

    from copy import copy
    # algs = [REPS, REPS_MI]
    # params = [{'eps': 0.5}, {'eps': 0.5, 'k': 2}]

    dim = 1

    # algs = [REPS_MI, REPS]
    # params = [{'eps': 0.5, 'k': 2}, {'eps': 0.5}]

    # algs = [REPS]
    # params = [{'eps': 0.2}]

    algs = [REPS, REPS, REPS]
    params = [{'eps': 0.3}, {'eps': 0.2}, {'eps': 0.15}, {'eps': 0.1}, {'eps': 0.05}]

    return_mean_algos = []
    return_std_algos = []
    mis_algos = []


    for alg, param in zip(algs, params):

        
        return_mean_algo = []
        return_std_algo = []
        mis_algo = []

        for _ in range(5):

            # MDP
            mdp = LQR.generate(dimensions=dim, episodic=True)#, max_pos=1., max_action=1.)

            optimal_control(mdp)

            # optimal_control(mdp)

            # mdp.Q[1][1] = 0
            # mdp.R[0][0] = 0.1
            # mdp.R[1][1] = 0
            
            # mdp.B[1][1] = 0
            # mdp.B[1][0] = 0

            # if dim > 2:
            #     mdp.Q[2][2] = 0
            #     mdp.R[2][2] = 0
            #     mdp.B[2][2] = 0

            # if dim > 3:
            #     mdp.Q[3][3] = 0
            #     mdp.R[3][3] = 0
            #     mdp.B[3][3] = 0

            print('Q', mdp.Q)
            print('R', mdp.R)
            print('B', mdp.B)

            n_epochs = 50 # 100
            ep_per_fit = 100 # 100
            fit_per_epoch = 10
            return_mean, return_std, agent = experiment(copy(alg), copy(mdp), param, n_epochs=n_epochs, fit_per_epoch=fit_per_epoch, ep_per_fit=ep_per_fit)

            return_mean_algo += [return_mean]
            return_std_algo += [return_std]
            if hasattr(agent, 'mis'):
                mis_algo += [agent.mis]

        return_mean_algos += [return_mean_algo]
        return_std_algos += [return_std_algo]
        if hasattr(agent, 'mis'):
            mis_algos += [mis_algo]

    optimal_control(mdp)

    fig, ax = plt.subplots()

    y = np.array(return_mean_algos).mean(axis=1)
    x = np.arange(0, y.shape[1], 1)
    ci = np.array(return_std_algos).mean(axis=1)

    for i in range(y.shape[0]):
        ax.plot(x,y[i])
        ax.fill_between(x, (y-ci)[i], (y+ci)[i], alpha=.3)

    plt.title(f'samples: {ep_per_fit}')
    # plt.legend([alg.__name__ for alg in algs])
    plt.legend([str(p['eps']) for p in params])
    plt.savefig(f'imgs/samples_{ep_per_fit}_dim_{dim}_eps_{str(param["eps"]).replace(".", "_")}.png')

    fig, ax = plt.subplots()

    y = np.array(mis_algos).mean(axis=1)
    x = np.arange(0, y.shape[1], 1)
    ci = np.array(mis_algos).std(axis=1)[0]
    y = y[0]

    for i in range(y.shape[1]):
        ax.plot(x,y[:,i])
        ax.fill_between(x, (y-ci)[:,i], (y+ci)[:,i], alpha=.3)

    plt.title(f'MI for parameters w/ {ep_per_fit} samples')
    plt.legend(list(range(y.shape[1])))
    plt.savefig(f'imgs/samples_{ep_per_fit}_mi_dim_{dim}.png')