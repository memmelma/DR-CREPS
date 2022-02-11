import numpy as np

from scipy.optimize import minimize

from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization

from .utils import compute_corr

from distributions import GaussianDistributionGDR, GaussianDiagonalDistribution

class DR_REPS_PE(BlackBoxOptimization):
	"""
	Episodic Relative Entropy Policy Search algorithm with Guided Dimensionality Reduction, and Prioritized Exploration.
	Based on "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.
	"""
	def __init__(self, mdp_info, distribution, policy, eps, lambd, k, gdr=True, C='MI', mi_estimator='regression', features=None):
		"""
		Constructor.

		Args:
			eps ([float, Parameter]): the maximum admissible value for the Kullback-Leibler
				divergence between the new distribution and the
				previous one at each update step.
			lambd ([float, Parameter]): the discount factor to reduce exploration in ineffective parameters 0 \leq \lambda \leq 1. 
				\lambda = 1 is equivalent to no Prioritized Exploration (PE). 
			k ([float, Parameter]): number of effective parameters to choose. k = number of parameters is equal to no Guided
				Dimensionality Reduction (GDR). However, C is estimated in rotated space.
			gdr ([bool]): whether to use guided dimensionality reduction or diagonal covariance matrix.
			C ([str]): corrleation measure. One of 'PCC' (pearson correlation coefficient) 
													or 'MI' (mutual information).
			mi_estimator ([str]) MI estimator. One of 'regression' (sklearn.feature_selection.mutual_info_regression) 
													or 'score' (sklearn.metrics.mutual_info_score)
													or 'hist' (histogram based estimation from samples)
		"""

		self._eps = to_parameter(eps)
		self._lambd = to_parameter(lambd)
		self._k = to_parameter(k)

		self._mi_estimator = mi_estimator

		self.corr_list = []
		self.C = C
		self.gdr = gdr
		
		self._add_save_attr(_eps='mushroom')
		self._add_save_attr(_kappa='mushroom')
		self._add_save_attr(_lambd='mushroom')
		self._add_save_attr(_k='mushroom')

		super().__init__(mdp_info, distribution, policy, features)
	
		if self.gdr:
			assert type(self.distribution) is GaussianDistributionGDR, f'Only GaussianDistributionGDR supports GDR, got {type(self.distribution)}!'
		else:
			assert type(self.distribution) is GaussianDiagonalDistribution, f'Only GaussianDiagonalDistribution supports PE w/o GDR, got {type(self.distribution)}!'


	def _update(self, Jep, theta):
		
		self.distribution._lambd = self._lambd()

		if type(self.distribution) is GaussianDistributionGDR:
			# rotate samples
			theta = ( self.distribution._u.T @ ( theta.T - self.distribution._mu[:,None] ) ).T

		# REPS
		eta_start = np.ones(1)
		res = minimize(DR_REPS_PE._dual_function, eta_start,
					   jac=DR_REPS_PE._dual_function_diff,
					   bounds=((np.finfo(np.float32).eps, np.inf),),
					   args=(self._eps(), Jep, theta))
		eta_opt = res.x.item()
		Jep -= np.max(Jep)
		d = np.exp(Jep / eta_opt)

		# compute correlation measure
		self.top_k_corr, corr = compute_corr(theta, Jep, k=self._k(), C=self.C, estimator=self._mi_estimator)
		
		# WMLE
		if type(self.distribution) is GaussianDistributionGDR:
			self.distribution.mle_gdr(theta, weights=d, indices=self.top_k_corr)
		elif type(self.distribution) is GaussianDiagonalDistribution:
			self.distribution.mle(theta, weights=d, indices=self.top_k_corr)
			self.distribution.update_importance(corr) #/np.sum(corr)


	@staticmethod
	def _dual_function(eta_array, *args):
		eta = eta_array.item()
		eps, Jep, theta = args

		max_J = np.max(Jep)

		r = Jep - max_J
		sum1 = np.mean(np.exp(r / eta))

		return eta * eps + eta * np.log(sum1) + max_J

	@staticmethod
	def _dual_function_diff(eta_array, *args):
		eta = eta_array.item()
		eps, Jep, theta = args

		max_J = np.max(Jep)

		r = Jep - max_J

		sum1 = np.mean(np.exp(r / eta))
		sum2 = np.mean(np.exp(r / eta) * r)

		gradient = eps + np.log(sum1) - sum2 / (eta * sum1)

		return np.array([gradient])
