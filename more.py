from scipy.optimize import minimize

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features.basis.polynomial import PolynomialBasis

import autograd.numpy as np
from autograd import grad

import traceback

class MORE(BlackBoxOptimization):
    """
    "Model-Based Relative Entropy Stochastic Search"
    Abdolmaleki, Abbas & Lioutikov, Rudolf & Peters, Jan & Lau, Nuno & Reis, Luís & Neumann, Gerhard.
    """
    def __init__(self, mdp_info, distribution, policy, eps, beta, features=None):
        """
        Constructor.

        Args:
            eps (float): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.
            beta (float): the upper bound for the entropy of the new policy.

        """
        self.eps = eps
        self.beta = beta

        self._add_save_attr(eps='primitive')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):

        # get distribution parameters mu and sigma
        n = len(self.distribution._mu)
        dist_params = self.distribution.get_parameters()
        mu_t = dist_params[:n][:,np.newaxis]
        chol_sig_empty = np.zeros((n,n))
        chol_sig_empty[np.tril_indices(n)] = dist_params[n:]
        sig_t = chol_sig_empty.dot(chol_sig_empty.T)

        # set beta as specified in paper -> MORE page 4
        gamma = 0.99
        entropy_policy_min = -75
        entropy_policy = MORE.closed_form_entropy(sig_t, n)
        self.beta = gamma * (entropy_policy - entropy_policy_min) + entropy_policy_min
        # print('\nbeta', self.beta)

        # create polynomial features
        poly_basis = PolynomialBasis().generate(2, theta.shape[1])
        features = np.array([[poly(t_i) for poly in poly_basis] for t_i in theta])

        # fit linear regression
        regressor = Regressor(LinearApproximator,
                      input_shape=(features.shape[1],),
                      output_shape=(1,))
        regressor.fit(features, Jep)

        # get quadratic surrogate model from learned regression
        R, r, r_0 = MORE.get_quadratic_model_from_regressor(regressor, theta) # MORE.get_surrogate(theta, Jep)

        # MORE lagrangian -> bounds from MORE page 3
        eta_omg_start = np.ones(2)
        res = minimize(MORE._dual_function, eta_omg_start,
                       jac=grad(MORE._dual_function),
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(self.eps, self.beta, sig_t, mu_t, R, r, Jep, theta),
                       method=None)
        eta_opt, omg_opt = res.x[0], res.x[1]
        if not res.success:
            print('\neta_opt', eta_opt, 'omg_opt', omg_opt, 'optimizer success', res)

        ## MLE update using mushroom -> REPS Compendium page 17 (between equations 76 and 77)

        # get parameters for MLE
        R_theta = regressor.predict(features)
        d = np.exp(R_theta / (eta_opt+omg_opt)).squeeze()

        pwr = (eta_opt/(eta_opt+omg_opt))
        theta_pwr = np.sign(theta) * (np.abs(theta)) ** pwr

        # update distribution
        self.distribution.mle(theta_pwr, d)
        return

        ## MLE update using closed form solution -> REPS Compendium page 17 equation 77

        # calculate closed form solution
        mu_t1, sig_t1 = MORE.closed_form_mu_t1_sig_t1(sig_t, mu_t, R, r, eta_opt, omg_opt)

        try:
            dist_params = np.concatenate((mu_t1.flatten(), np.linalg.cholesky(sig_t1)[np.tril_indices(n)].flatten()))
        except np.linalg.LinAlgError:
            traceback.print_exc()
            print('error in setting dist_params - sig_t1 not positive definite')
            print('mu_t1', mu_t1)
            print('sig_t1', sig_t1)
            print('eta_opt', eta_opt, 'omg_opt', omg_opt)
            exit(42)

        # update distribution
        self.distribution.set_parameters(dist_params)

    @staticmethod
    def _dual_function(lag_array, *args):
        eta, omg = lag_array[0], lag_array[1]
        eps, beta, Q, b, R, r, Jep, theta = args

        F, f = MORE.get_F_f(Q, b, R, r, eta)
        
        term1 = (f.T @ F @ f) - eta * (b.T @ np.linalg.inv(Q) @ b) - eta * (np.linalg.slogdet(2*np.pi*Q)[1]) + (eta + omg) * (np.linalg.slogdet(2*np.pi*(eta + omg) * F)[1])

        return eta * eps - beta * omg + 0.5 * term1

    @staticmethod
    def closed_form_mu_t1_sig_t1(*args):
        # sig, mu, R, r, eta, beta
        Q, b, R, r, eta, omg = args

        F, f = MORE.get_F_f(Q, b, R, r, eta)

        mu_t1 = F @ f
        sig_t1 = F * (eta + omg)

        return mu_t1, sig_t1

    @staticmethod
    def get_F_f(*args):
        Q, b, R, r, eta = args
        F = np.linalg.inv(eta * np.linalg.inv(Q) - 2 * R)
        f = eta * np.linalg.inv(Q) @ b + r

        return F, f
    
    @staticmethod
    def get_quadratic_model_from_regressor(regressor, theta):

        # get parameter vector beta
        beta = regressor.get_weights()

        # reconstruct components for quadratic surrogate model from beta
        r_0 = beta[0]
        r = beta[1:theta.shape[1]+1][:,np.newaxis]

        R = np.zeros((theta.shape[1],theta.shape[1]))
        beta_ctr = 1 + theta.shape[1]  
        for i in range(theta.shape[1]):
            for j in range(i, theta.shape[1]):
                if i == j:
                    R[i][j] = beta[beta_ctr]
                else:
                    R[i][j] = beta[beta_ctr]/2
                    R[j][i] = beta[beta_ctr]/2
                beta_ctr += 1

        # verification that components got reconstructed correctly 
        # theta_pred = np.atleast_2d(theta[0])
        # quadratic_pred = theta_pred @ R @ theta_pred.T + theta_pred @ r + r_0
        # regressor_pred = regressor.predict(features)[0]
        # print('equal?', quadratic_pred == regressor_pred, 'quadratic model', quadratic_pred, 'regressor', regressor_pred)

        return R, r, r_0


    # calculate entropy of policy
    @staticmethod
    def closed_form_entropy(*args):
        '''
        Taken from page 11 of REPS compendium
        '''
        sig, n = args
        logdet_sig, _ = np.linalg.slogdet(sig)
        c = MORE.get_c(n)
        return 0.5 * (logdet_sig + c + n)
    
    @staticmethod
    def get_c(args):
        n = args
        return n * np.log(2*np.pi)


    # manual polynomial regression (not in use)
    @staticmethod
    def get_phi(theta):
        # bias term
        phi = np.ones(theta.shape[0])[:, np.newaxis]
        # linear terms
        phi = np.concatenate((phi, theta), axis=1)
        # quadratic terms
        for i in range(theta.shape[1]):
            for j in range(i, theta.shape[1]):
                phi = np.concatenate((phi,(theta[:,i]*theta[:,j])[:, np.newaxis]), axis=1)
        
        return phi
    
    @staticmethod
    def get_prediction(theta, y):
        beta = MORE.get_linear_regression_beta(theta, y)
        phi = MORE.get_phi(theta)
        y_hat = phi @ beta
        return y_hat
    
    @staticmethod
    def get_linear_regression_beta(theta, y):
        print(y.shape)
        phi = MORE.get_phi(theta)
        beta = np.linalg.inv(phi.T @ phi) @ phi.T @ y
        print(beta.shape)
        return beta[:, np.newaxis]
