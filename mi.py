import numpy as np
np.random.seed(42)

def entropy(sig):
    n = sig.shape[0]
    return 0.5*np.linalg.slogdet(sig)[1] + (np.log(2*np.pi))/2 + n/2

# TODO: only analytical for multivariate so far
dim = 1

# p(X)
mu_x = np.atleast_1d(np.random.rand(dim))
sig_x = np.atleast_2d(np.random.rand(dim,dim))

# linear transformation matrix A
A = np.atleast_2d(np.random.rand(dim,dim))

# noise distribution E
# mu_e = np.atleast_1d(np.random.rand(dim))
# sig_e = np.atleast_2d(np.random.rand(dim,dim))
mu_e = np.atleast_1d(np.zeros(dim))
sig_e = np.atleast_2d(np.zeros((dim,dim))) * 1e-3

# p(Y|X)
mu_y_x = A @ mu_x + mu_e
sig_y_x = A @ sig_x @ A.T + sig_e

# p(Y)
# TODO: wrong
const = 1/np.sqrt(np.linalg.det(2*np.pi*(sig_x + sig_y_x))) * np.exp(-1/2 * (mu_x - mu_y_x).T @ np.linalg.inv(sig_x + sig_y_x) @ (mu_x - mu_y_x))
sig_y = np.linalg.inv( np.linalg.inv(sig_x) + np.linalg.inv(sig_y_x) ) * const
# sig_y = ( np.linalg.inv(sig_x) + np.linalg.inv(sig_y_x) ) # * const
# sig_y = sig_x@np.linalg.inv(sig_x + sig_y_x)@sig_y_x

# p(X,Y)
sig_xy = np.block( [ [sig_x,    sig_x@A.T], 
                    [A@sig_x,   sig_y_x + A@sig_x@A.T + sig_e] ] )
    
H_x = entropy(sig_x)
H_y_x = entropy(sig_y_x)
H_y = entropy(sig_y)
H_xy = entropy(sig_xy)
# TODO: H(y) >= H(y|x)
print('H_x', H_x, 'H_y_x', H_y_x, 'H_y', H_y, 'H_xy', H_xy)

# TODO: both should be equal
I_xy = H_x + H_y - H_xy
print('analytical', I_xy)
I_xy = H_y - H_y_x
print('analytical', I_xy)

n_samples = 1000
x = np.random.multivariate_normal(mu_x, sig_x, size=n_samples).T
e = np.random.multivariate_normal(mu_e, sig_e, size=n_samples).T

y = A@x + e

from sklearn.feature_selection import mutual_info_regression
def calc_MI_sklearn_regression(x, y, bins):
    mi = mutual_info_regression(x,y.squeeze())
    return mi.squeeze()

# https://stackoverflow.com/a/20505476

from scipy.stats import chi2_contingency
def calc_MI_scipy(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi

from sklearn.metrics import mutual_info_score
def calc_MI_sklearn(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

bins = 10

I_xy_sklearn_regression = calc_MI_sklearn_regression(x.T, y.T, bins)
I_xy_sklearn = calc_MI_sklearn(x.squeeze(), y.squeeze(), bins)
I_xy_scipy = calc_MI_scipy(x.squeeze(), y.squeeze(), bins)

print('sklearn_regression', I_xy_sklearn_regression, 'sklearn', I_xy_sklearn, 'scipy', I_xy_scipy)