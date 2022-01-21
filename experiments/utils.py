import numpy as np

from algorithms import DR_CREPS_PE, DR_REPS_PE, RWR_PE
from distributions import GaussianDiagonalDistribution, GaussianDistributionGDR
from mushroom_rl.distributions import GaussianDistribution, GaussianCholeskyDistribution

from mushroom_rl.distributions.distribution import Distribution
from mushroom_rl.algorithms.policy_search.black_box_optimization import REPS, RWR, ConstrainedREPS, MORE

def init_distribution(mu_init=0, sigma_init=1e-3, size=1, sample_strat=None, lambd=0.0, distribution_class='diag'):
    
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

    assert sample_strat is not None and (distribution_class != 'cholesky' or distribution_class != 'fixed'), \
        f"Argument 'sample_strat' only supported for distribution_class = 'diag' or 'gdr', got {distribution_class}!"
    if sample_strat is not None:
        distribution.set_sample_strat(sample_strat=sample_strat, lambd=lambd)

    return distribution


def init_algorithm(algorithm_class='REPS', params={}):

    if algorithm_class == 'REPS':
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
