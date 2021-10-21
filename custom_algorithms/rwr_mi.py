import numpy as np

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.utils.parameters import to_parameter

from mushroom_rl.utils.parameters import ExponentialParameter, LinearParameter, Parameter

from sklearn.feature_selection import mutual_info_regression

from scipy.stats import pearsonr

class RWR_MI(BlackBoxOptimization):
    """
    Reward-Weighted Regression algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, mdp_info, distribution, policy, eps, gamma, k, bins, mi_type='regression', method='MI', mi_avg=False, oracle=None, features=None):
        """
        Constructor.

        Args:
            beta ([float, Parameter]): the temperature for the exponential reward
                transformation.

        """
        self._beta = to_parameter(eps)

        self._add_save_attr(_beta='mushroom')

        self._k = to_parameter(k)
        print('K', k, self._k())
        
        self._bins = to_parameter(bins)

        self._mi_type = mi_type
        self._mi_avg = mi_avg
        self.method = method

        self.mis = []
        self.mi_avg = np.zeros(len(distribution._mu))
        self.alpha = ExponentialParameter(1, exp=0.5)
        
        self.mus = []
        self.kls = []
        self.entropys = []
        
        self.top_k_mis = []

        self.oracle = oracle

        if gamma == -1:
            print('Using LinearParameter 1->0')
            self.beta = LinearParameter(0., threshold_value=1., n=50)
            # self.beta = LinearParameter(0., threshold_value=1., n=100)
        elif gamma == -2:
            print('Using LinearParameter 0->1')
            self.beta = LinearParameter(1., threshold_value=0., n=100)
        else:
            self.beta = Parameter(1-gamma)

        self.samples_theta = []
        self.samples_Jep = []

        super().__init__(mdp_info, distribution, policy, features)

    def compute_mi(self, theta, Jep, type='regression'):
        print('computing MI w/', theta.shape, Jep.shape)
        if type == 'score':
            from sklearn.metrics import mutual_info_score
            mi = []
            for theta_i in theta.T:
                        c_xy = np.histogram2d(theta_i, Jep, self._bins())[0]
                        mi += [mutual_info_score(None, None, contingency=c_xy)]
            mi = np.array(mi)
        elif type == 'regression':
            mi = mutual_info_regression(theta, Jep, discrete_features=False, n_neighbors=self._bins(), random_state=42)
            print(mi.shape)
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
        if len(self.samples_theta) == 0:
            self.samples_theta = theta
            self.samples_Jep = Jep
        else:
            self.samples_theta = np.concatenate((self.samples_theta,theta))
            self.samples_Jep = np.concatenate((self.samples_Jep,Jep))

        self.distribution._gamma = 1 - self.beta()
        Jep -= np.max(Jep)

        d = np.exp(self._beta() * Jep)

        if self.method == 'MI':
            mi = self.compute_mi(theta, Jep, type=self._mi_type)
        elif self.method == 'Pearson':
            mi = self.compute_pearson(theta, Jep)
        elif self.method == 'MI_ALL':
            mi = self.compute_mi(self.samples_theta, self.samples_Jep, type=self._mi_type)

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

        self.top_k_mis += [top_k_mi]

        self.distribution.mle(theta, d, top_k_mi)

        importance = self.mi_avg #/ np.sum(self.mi_avg)
        self.distribution.update_importance(importance)

        self.mus += [0]
        self.kls += [0]
        self.entropys += [0]