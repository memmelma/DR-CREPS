import numpy as np
np.random.seed(42)

def entropy(sig):
    n = sig.shape[0]
    return n/2 *(np.log(2*np.pi) + 1) + 1/2*np.linalg.slogdet(sig)[1] 

dim = 1

mu_x = np.random.rand(dim) 
sig_x = np.random.rand(dim,dim)

A = np.random.rand(dim,dim)
A = np.ones((dim,dim))
b = np.random.normal(size=dim)

mu_y = A@mu_x + b
sig_y = A@sig_x@A.T

# sig_xy = sig_x@np.linalg.inv(sig_x+sig_y)@sig_y
sig_xy = np.linalg.inv(sig_x) + np.linalg.inv(sig_y)
    
H_x = entropy(sig_x)
H_y = entropy(sig_y)
H_xy = entropy(sig_xy)

# I_xy = H_x + H_y - H_xy

I_xy = H_xy - H_y
print('analytical', I_xy)

n_samples = 1000
x = np.random.multivariate_normal(mu_x, sig_x, size=n_samples)
b = np.random.normal(size=n_samples)

y = A@x.T + b

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
I_xy_sklearn_regression = calc_MI_sklearn_regression(x, y, bins)
I_xy_sklearn = calc_MI_sklearn(x.squeeze(), y.squeeze(), bins)
I_xy_scipy = calc_MI_scipy(x.squeeze(), y.squeeze(), bins)
print('sklearn_regression', I_xy_sklearn_regression, 'sklearn', I_xy_sklearn, 'scipy', I_xy_scipy)