import numpy as np

from custom_distributions.gaussian_custom import GaussianDiagonalDistribution, GaussianCholeskyDistribution, GaussianDistribution, GaussianDistributionMI

from mushroom_rl.algorithms.policy_search.black_box_optimization import REPS, RWR
from custom_algorithms.more import MORE
from custom_algorithms.reps_mi import REPS_MI
from custom_algorithms.rwr_mi import RWR_MI
from custom_algorithms.constrained_reps import ConstrainedREPS
from custom_algorithms.constrained_reps_mi import ConstrainedREPSMI

from custom_algorithms.reps_mi_full import REPS_MI_full
from custom_algorithms.constrained_reps_mi_full import ConstrainedREPSMIFull

def init_distribution(mu_init=0, sigma_init=1e-3, size=1, sample_type=None, gamma=0.0, distribution_class='diag'):
    
    mu = mu_init * np.ones(size)

    if distribution_class == 'diag':

        # load full covariance
        if type(sigma_init) != float:
            distribution = sigma_init
            print('Successfully loaded distribution!')
        else:
            sigma = sigma_init * np.ones(size)
            distribution = GaussianDiagonalDistribution(mu, sigma)

        if sample_type == 'fixed':
            distribution.set_fixed_sample(gamma=gamma)
        elif sample_type == 'percentage':
            distribution.set_percentage_sample(gamma=gamma)
        elif sample_type == 'importance':
            distribution.set_importance_sample()
        elif sample_type == 'PRO':
            distribution.set_PRO_sample()

    elif distribution_class == 'cholesky':
        print('sample_type only supported for diagonal covariance')
        print('sigma_init passed is std -> cov = sigma_init**2')
        sigma = sigma_init**2 * np.eye(size)
        print(sigma)
        distribution = GaussianCholeskyDistribution(mu, sigma)
    elif distribution_class == 'fixed':
        print('sample_type only supported for diagonal covariance')
        print('sigma_init passed is std -> cov = sigma_init**2')
        sigma = sigma_init**2 * np.eye(size)
        distribution = GaussianDistribution(mu, sigma)
    elif distribution_class == 'mi':
        sigma = sigma_init**2 * np.eye(size)
        distribution = GaussianDistributionMI(mu, sigma)

        if sample_type == 'fixed':
            distribution.set_fixed_sample(gamma=gamma)
        elif sample_type == 'percentage':
            distribution.set_percentage_sample(gamma=gamma)
        elif sample_type == 'importance':
            distribution.set_importance_sample()
        elif sample_type == 'PRO':
            distribution.set_PRO_sample()

    return distribution

def init_algorithm(algorithm_class='REPS', params={}):

    # algorithms
    if algorithm_class == 'REPS':
        alg = REPS
        params = {'eps': params['eps']}

    elif algorithm_class == 'REPS_MI_full':
        alg = REPS_MI_full
        params = {'eps': params['eps'], 'gamma': params['gamma'], 'k': params['k'], 'bins': params['bins'],  'method': params['method'],'mi_type': params['mi_type'], 'mi_avg': params['mi_avg']}

    elif algorithm_class == 'REPS_MI':
        alg = REPS_MI
        params = {'eps': params['eps'], 'gamma': params['gamma'], 'k': params['k'], 'bins': params['bins'], 'method': params['method'], 'mi_type': params['mi_type'], 'mi_avg': params['mi_avg']}
      
    elif algorithm_class == 'REPS_MI_ORACLE':
        alg = REPS_MI
        params = {'eps': params['eps'], 'gamma': params['gamma'], 'k': params['k'], 'bins': params['bins'], 'method': params['method'], 'mi_type': params['mi_type'], 'mi_avg': params['mi_avg'], 'oracle': params['oracle']}

    elif algorithm_class == 'RWR':
        alg = RWR
        params = {'beta': params['eps']}
    
    elif algorithm_class == 'RWR_MI':
        alg = RWR_MI
        params = {'eps': params['eps'], 'gamma': params['gamma'], 'k': params['k'], 'bins': params['bins'], 'method': params['method'], 'mi_type': params['mi_type'], 'mi_avg': params['mi_avg']}

    elif algorithm_class == 'MORE':
        alg = MORE
        params = {'eps': params['eps'], 'kappa': params['kappa']}
        
    elif algorithm_class == 'ConstrainedREPS':
        alg = ConstrainedREPS
        params = {'eps': params['eps'], 'kappa': params['kappa']}

    elif algorithm_class == 'ConstrainedREPSMI':
        alg = ConstrainedREPSMI
        params = {'eps': params['eps'], 'k': params['k'], 'kappa': params['kappa'], 'gamma': params['gamma'], 'bins': params['bins'], 'method': params['method'], 'mi_type': params['mi_type'], 'mi_avg': params['mi_avg']}

    elif algorithm_class == 'ConstrainedREPSMIFull':
        alg = ConstrainedREPSMIFull
        params = {'eps': params['eps'], 'k': params['k'], 'kappa': params['kappa'], 'gamma': params['gamma'], 'bins': params['bins'], 'method': params['method'], 'mi_type': params['mi_type'], 'mi_avg': params['mi_avg']}
    
    elif algorithm_class == 'ConstrainedREPSMIOracle':
        alg = ConstrainedREPSMI
        params = {'eps': params['eps'], 'k': params['k'], 'kappa': params['kappa'], 'gamma': params['gamma'], 'bins': params['bins'], 'method': params['method'], 'mi_type': params['mi_type'], 'mi_avg': params['mi_avg'], 'oracle': params['oracle']}

    return alg, params

def log_constraints(agent):
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
    
    return mus, kls, entropys, mi_avg