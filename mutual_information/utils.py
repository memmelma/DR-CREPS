import numpy as np
from numpy.random import default_rng
import scipy.stats as st
from scipy.stats import pearsonr
from knnie import kraskov_mi, revised_mi
from sklearn.feature_selection import mutual_info_regression

def get_mean_and_confidence(data):
	# Code taken from https://github.com/MushroomRL/mushroom-rl-benchmark/blob/28872e5d09e9afabba0ece8cbf827d296d427af4/mushroom_rl_benchmark/utils/plot.py
    """
    Compute the mean and 95% confidence interval
    Args:
        data (np.ndarray): Array of experiment data of shape (n_runs, nepochs).
    Returns:
        The mean of the dataset at each epoch along with the confidence interval.
    """
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)
    interval  = st.t.interval(0.95, n - 1, scale=se)
    return mean, interval

def get_style():
	legend = ['$MI_{true}$', '$MI_{regression}$', '$MI_{histogram}$', '$MI_{KSG}$']
	colors = ['#CC3311', '#EE3377', '#0077BB', '#009988']
	linestyles = ['-', ':', '--', '-.', 'solid', 'dashed', 'dashdot', 'dotted']

	return legend, colors, linestyles
	
def entropy(sig):
	n = sig.shape[0]
	return 0.5*np.linalg.slogdet(sig)[1] + (n*np.log(2*np.pi))/2 + n/2

def shan_entropy(c):
	c_normalized = c / float(np.sum(c))
	c_normalized = c_normalized[np.nonzero(c_normalized)]
	H = - np.sum( c_normalized * np.log( c_normalized ) )
	return H
	
def calc_MI_sklearn_regression(x, y, n_neighbors=3, random_state=None):
	# KSG implementation of sklearn (uses standardization of input)
	# https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/feature_selection/_mutual_info.py#L291
	return mutual_info_regression(x, y.ravel(), discrete_features=False, n_neighbors=n_neighbors, random_state=random_state)[0]
	
def calc_MI_KSG(x, y, n_neighbors=3):
	# "It is (KSG)’s inability to handle noise that diminishes it’s effectiveness in real data sets." - On the Estimation of Mutual Information
	return kraskov_mi(x, y, k=n_neighbors)

def calc_MI_revised_KSG(x, y, n_neighbors=3):
	return revised_mi(x, y, k=n_neighbors)

def calc_MI_samples(x, y, bins):
	# https://stackoverflow.com/a/20505476
	c_X = np.histogram(x.squeeze(), bins=bins)[0]
	c_Y = np.histogram(y, bins=bins)[0]
	c_XY = np.histogram2d(x.squeeze(), y, bins=bins)[0]
	H_X = shan_entropy(c_X)
	H_Y = shan_entropy(c_Y)
	H_XY = shan_entropy(c_XY)
	return H_X + H_Y - H_XY

def calc_PCC(x,y):
	PCC = 0
	for i in range(x.shape[1]):
		PCC += pearsonr(x[:,i], y)[0]
	return PCC

def analytical_MI(m, n, samples, random_seed):

	# fixed rng for p(X)
	fixed_rng = default_rng(0)
	# seeded rng for p(E), A, X~p(X), E~p(E)
	seed_rng = default_rng(random_seed)

	# p(X)
	mu_x = np.atleast_1d(fixed_rng.random(m))
	sig_x = np.atleast_2d(fixed_rng.random((m,m)))
	if m > 1:
		sig_x = sig_x @ sig_x.T

	# non zero noise
	mu_e = np.atleast_1d(seed_rng.random(n))
	sig_e = np.atleast_2d(seed_rng.random((n,n)))
	if n > 1:
		sig_e = sig_e @ sig_e.T

	# zero noise -> determinisitic function f(X) = Y -> https://stats.stackexchange.com/questions/465056/mutual-information-between-x-and-fx
	# mu_e = np.zeros(n)
	# sig_e = np.zeros((n,n))

	# linear transformation matrix A (full rank)
	A = np.atleast_2d(seed_rng.random((n,m)))
	if n == m and n > 1 and m > 1:
		A = A @ A.T

	# p(Y|X)
	# https://ssl2.cms.fu-berlin.de/ewi-psy/einrichtungen/arbeitsbereiche/computational_cogni_neurosc/PMFN/10-Gaussian-distributions.pdf
	# eq. 10.20
	mu_y_x = A @ mu_x + mu_e
	sig_y_x = A @ sig_x @ A.T + sig_e

	# p(Y)
	# Bishop p.93 eq. 2.115
	mu_y = A @ mu_x
	sig_y = sig_y_x + A @ sig_x @ A.T

	# p(X|Y)
	# Bishop p.93 eq. 2.116
	sig_x_y = np.linalg.inv(np.linalg.inv(sig_x) + A.T @ np.linalg.inv(sig_y_x) @ A)
	mu_x_y = sig_x_y @ (A.T @ np.linalg.inv(sig_y_x) @ (mu_y - mu_e) + np.linalg.inv(sig_x) @ mu_x)

	# p(X,Y)
	# https://ssl2.cms.fu-berlin.de/ewi-psy/einrichtungen/arbeitsbereiche/computational_cogni_neurosc/PMFN/10-Gaussian-distributions.pdf
	# eq. 10.26 : \sigma_{yy} is covariance of p(Y|X) !
	mu_xy = np.concatenate([mu_x, A @ mu_x + mu_e])
	sig_xy = np.block( [[sig_x,	sig_x @ A.T], 
						[A @ sig_x,   sig_y_x + A @ sig_x @ A.T]] )

	H_x, H_y, H_xy, H_y_x, H_x_y = entropy(sig_x), entropy(sig_y), entropy(sig_xy), entropy(sig_y_x), entropy(sig_x_y)
	
	I_0 = H_y - H_y_x
	I_1 = H_x - H_x_y
	I_2 = H_xy - H_x_y - H_y_x
	I_3 = H_x + H_y - H_xy
	
	assert np.round(I_0,2) == np.round(I_1,2) and np.round(I_1,2) == np.round(I_2,2) and np.round(I_2,2) == np.round(I_3,2), \
		f'all MI formulations should yield the same result! got: {I_0} {I_1} {I_2} {I_3}'

	X = seed_rng.multivariate_normal(mu_x, sig_x, size=samples)
	E = seed_rng.multivariate_normal(mu_e, sig_e, size=samples)

	Y = (A@X.T).T + E

	return X, Y, I_1

def compute_MI(x, y, I_gt, bins, random_seed):

	I_xy_sklearn_regression, I_xy_samples, I_xy_ksg= [], [], []

	for i in range(y.shape[1]):
		x_tmp = x
		y_tmp = y[:,i]
		for bin_i, bin in enumerate(bins):
			if i == 0:
				I_xy_samples += [calc_MI_samples(x_tmp, y_tmp, bin)]
				I_xy_ksg += [calc_MI_KSG(x_tmp, y_tmp[:,None], bin)]
				I_xy_sklearn_regression += [calc_MI_sklearn_regression(x, y_tmp, n_neighbors=bin, random_state=random_seed)]
			else:
				I_xy_samples[bin_i] += calc_MI_samples(x_tmp, y_tmp, bin)
				I_xy_ksg[bin_i] += calc_MI_KSG(x_tmp, y_tmp[:,None], bin)
				I_xy_sklearn_regression[bin_i] += [calc_MI_sklearn_regression(x, y_tmp, n_neighbors=bin, random_state=random_seed)]

	return [I_gt, *I_xy_sklearn_regression, *I_xy_samples, *I_xy_ksg]


