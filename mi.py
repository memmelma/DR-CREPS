import numpy as np
np.random.seed(42)

# closed form entropy as in REPS compendium 
def entropy(sig):
    n = sig.shape[0]
    return 0.5*np.linalg.slogdet(sig)[1] + (n*np.log(2*np.pi))/2 + n/2

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
# https://ssl2.cms.fu-berlin.de/ewi-psy/einrichtungen/arbeitsbereiche/computational_cogni_neurosc/PMFN/10-Gaussian-distributions.pdf eq. 10.20
mu_y_x = A @ mu_x + mu_e
sig_y_x = A @ sig_x @ A.T + sig_e

# p(Y)
# Bishop p.93 eq. 2.115
sig_y = sig_y_x + A@sig_x@A.T
# p(X|Y)
# Bishop p.93 eq. 2.116
sig_x_y = np.linalg.inv(np.linalg.inv(sig_x) + A.T@np.linalg.inv(sig_y_x)@A)

# p(X,Y)
sig_xy = np.block( [ [sig_x,    sig_x@A.T], 
                    [A@sig_x,   sig_y_x + A@sig_x@A.T] ] )
    
H_x = entropy(sig_x)
H_y = entropy(sig_y)
H_xy = entropy(sig_xy)
H_y_x = entropy(sig_y_x)
H_x_y = entropy(sig_x_y)

print('H_x', H_x, 'H_y', H_y, 'H_y_x', H_y_x, 'H_x_y', H_x_y, 'H_xy', H_xy)

I_xy = H_x + H_y - H_xy
print('H_x + H_y - H_xy', I_xy)
I_xy = H_y - H_y_x
print('H_y - H_y_x', I_xy)
I_xy = H_x - H_x_y
print('H_x - H_x_y', I_xy)
I_xy = H_xy - H_x_y - H_y_x
print('H_xy - H_x_y - H_y_x', I_xy)

n_samples = 500
x = np.random.multivariate_normal(mu_x, sig_x, size=n_samples).T
e = np.random.multivariate_normal(mu_e, sig_e, size=n_samples).T

y = A@x + e

# https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/feature_selection/_mutual_info.py#L291
# uses NearestNeighbors and KDtree instead of contingency

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

print('sklearn_regression', I_xy_sklearn_regression, '\nsklearn', I_xy_sklearn, '\nscipy', I_xy_scipy)