import numpy as np

from scipy.optimize import minimize

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.utils.parameters import to_parameter

from mushroom_rl.utils.parameters import ExponentialParameter, LinearParameter, Parameter

from sklearn.feature_selection import mutual_info_regression

from scipy.stats import pearsonr

class REPS_MI(BlackBoxOptimization):
	"""
	Episodic Relative Entropy Policy Search algorithm with MI estension

	"""
	def __init__(self, mdp_info, distribution, policy, eps, gamma, k, bins, method='MI', mi_type='regression', mi_avg=True, oracle=None, features=None):
		"""
		Constructor.

		Args:
			eps ([float, Parameter]): the maximum admissible value for the Kullback-Leibler
				divergence between the new distribution and the
				previous one at each update step.
			kappa ([float, Parameter]): the maximum admissible value for the entropy decrease
				between the new distribution and the 
				previous one at each update step. 

		"""
		self._eps = to_parameter(eps)
		self._k = to_parameter(k)
		self._bins = to_parameter(bins)
		self._entropy_X = distribution.entropy() / len(distribution._mu)

		self._mi_type = mi_type
		self._mi_avg = mi_avg
		self.method = method

		self.mis = []
		self.mi_avg = np.zeros(len(distribution._mu))
		self.alpha = ExponentialParameter(1, exp=0.5)

		if gamma == -1:
			print('Using LinearParameter 1->0')
			# self.beta = LinearParameter(1-1e-10, threshold_value=0., n=100)
			# self.beta = LinearParameter(1., threshold_value=10., n=50)
			self.beta = LinearParameter(0., threshold_value=1., n=50)
		elif gamma == -2:
			print('Using LinearParameter 0->1')
			# self.beta = LinearParameter(1., threshold_value=0., n=100)
			self.beta = LinearParameter(10., threshold_value=1., n=100)
		else:
			self.beta = Parameter(1-gamma)

		self._add_save_attr(_eps='mushroom')
		self._add_save_attr(_kappa='mushroom')

		self.mus = []
		self.kls = []
		self.top_k_mis = []

		self.oracle = oracle

		super().__init__(mdp_info, distribution, policy, features)

	def compute_mi(self, theta, Jep, type='regression'):
		
		if type == 'score':
			from sklearn.metrics import mutual_info_score
			mi = []
			for theta_i in theta.T:
				c_xy = np.histogram2d(theta_i, Jep, self._bins())[0]
				mi += [mutual_info_score(None, None, contingency=c_xy)]
			mi = np.array(mi)
		elif type == 'regression':
			mi = mutual_info_regression(theta, Jep, discrete_features=False, n_neighbors=3, random_state=42)
		elif type == 'sample':
			mi = []
			for theta_i in theta.T:
				mi += [self.MI_from_samples(theta_i, Jep, self._bins())]
			mi = np.array(mi)

		return mi

	def compute_pearson(self, theta, Jep):
		p = []
		for i in range(theta.shape[1]):
			p += [pearsonr(theta[:,i], Jep)[0]]
			# p += [np.corrcoef(theta[:,i], Jep)[0][1]]
		p = np.abs(p)
		return np.nan_to_num(p)

	def MI_from_samples(self, x, y, bins):
		c_XY = np.histogram2d(x, y, bins)[0]
		# c_X = np.histogram(x, bins)[0]
		c_Y = np.histogram(y, bins)[0]

		# H_X = self.shan_entropy(c_X)
		H_Y = self.shan_entropy(c_Y)
		H_XY = self.shan_entropy(c_XY)
		MI = self._entropy_X + H_Y - H_XY
		return MI

	def shan_entropy(self, c):
		c_normalized = c / float(np.sum(c))
		c_normalized = c_normalized[np.nonzero(c_normalized)]
		H = -sum(c_normalized* np.log(c_normalized))  
		return H
	
	def _update(self, Jep, theta):
		
		self.distribution._gamma = 1 - self.beta()
		# self.distribution._gamma = np.log10(self.beta())
		# self.distribution._gamma = np.cbrt(self.beta())
		# self.distribution._gamma = np.sqrt(self.beta())
		# self.distribution._gamma = self.beta()**0.25
		# print(self.distribution._gamma)
		# REPS
		eta_start = np.ones(1)

		res = minimize(REPS_MI._dual_function, eta_start,
					   jac=REPS_MI._dual_function_diff,
					   bounds=((np.finfo(np.float32).eps, np.inf),),
					   args=(self._eps(), Jep, theta))

		eta_opt = res.x.item()

		Jep -= np.max(Jep)

		d = np.exp(Jep / eta_opt)
		
		
		if self.method == 'MI':
			mi = self.compute_mi(theta, Jep, type=self._mi_type)
		elif self.method == 'Pearson':
			mi = self.compute_pearson(theta, Jep)
		# TODO find better implt
		else:
			mi = self.compute_pearson(theta, Jep)

		if not self._mi_avg:
			self.mi_avg = mi / np.max((1e-18,np.max(mi)))
		else:
			self.mi_avg = self.mi_avg + self.alpha() * ( mi - self.mi_avg )
		self.mis += [self.mi_avg]
		
		if self._k() < 1:
			thresh = self.mi_avg.sum() * 0.2
			mi_sort = self.mi_avg.argsort()
			for i in range(len(self.mi_avg)+1):
				if self.mi_avg[ mi_sort[-(i+1):][::-1] ].sum() > thresh:
					top_mi = mi_sort[-(i+1):][::-1]
					break
			top_k_mi = top_mi
		else:
			top_k_mi = self.mi_avg.argsort()[-int(self._k()):][::-1]
		
		if self.oracle != None:
			top_k_mi = self.oracle

		print('top_k_mi',top_k_mi)
		self.top_k_mis += [top_k_mi]

		if self.method == 'Random':
			top_k_mi = np.random.randint(0, theta.shape[1], self._k())

		# self.distribution._top_k = top_k_mi
		self.distribution.mle(theta, d, indices=top_k_mi)

		importance = self.mi_avg #/ np.sum(self.mi_avg)
		self.distribution.update_importance(importance)

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