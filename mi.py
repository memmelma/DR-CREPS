import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn import neighbors
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression

import knnie

def entropy(sig):
	# closed form entropy as in REPS compendium
	n = sig.shape[0]
	return 0.5*np.linalg.slogdet(sig)[1] + (n*np.log(2*np.pi))/2 + n/2


def shan_entropy(c):
	c_normalized = c / float(np.sum(c))
	c_normalized = c_normalized[np.nonzero(c_normalized)]
	H = - np.sum( c_normalized * np.log( c_normalized ) )
	return H
	
def calc_MI_sklearn_regression(x, y, n_neighbors=3):
	# KSG implementation of sklearn
	# https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/feature_selection/_mutual_info.py#L291
	# Same except preprocessing (scaling of X, noise to X)
	# mi = mutual_info_regression(x, y, n_neighbors=n_neighbors, discrete_features=False)
	mi = knnie._estimate_mi(x, y.ravel(), n_neighbors=n_neighbors, discrete_features=False, discrete_target=False)
	return mi[0]

# "It is (KSG)’s inability to handle noise that diminishes it’s effectiveness in real data sets." - On the Estimation of Mutual Information
def calc_MI_KSG(x, y, n_neighbors=3):
	# KSG
	# mi = knnie.kraskov_mi(x, y, k=n_neighbors)
	# KSG from sklearn
	mi = knnie.kraskov_mi_sklearn(x, y, n_neighbors=n_neighbors)

	# KSG with prior
	# MI = knnie.kraskov_mi(x, y, H_X_given, k=n_neighbors)
	return mi

def calc_MI_samples(x, y, bins):
	
	# # https://stackoverflow.com/a/20505476
	# c_xy = np.histogram2d(x.ravel(), y.ravel(), bins)[0]
	# MI = mutual_info_score(None, None, contingency=c_xy)
	
	# # just samples
	x = x.ravel()
	y = y.ravel()
	
	c_X = np.histogram(x, bins=bins)[0]
	c_Y = np.histogram(y, bins=bins)[0]
	c_XY = np.histogram2d(x, y, bins=bins)[0]

	H_X = shan_entropy(c_X)
	H_Y = shan_entropy(c_Y)
	H_XY = shan_entropy(c_XY)
	
	MI = H_X + H_Y - H_XY

	return MI

def calc_MI_prior(x, y, bins, H_X_given, n):
	
	# # prior with samples
	# x = x.ravel()
	# y = y.ravel()
	
	# c_X = np.histogram(x, bins=bins)[0]
	# c_Y = np.histogram(y, bins=bins)[0]
	# c_XY = np.histogram2d(x, y, bins=bins)[0]

	# H_X = shan_entropy(c_X)
	# H_Y = shan_entropy(c_Y)
	# H_XY = shan_entropy(c_XY)
	
	# MI = H_X_given + H_Y - H_XY
	
	MI = knnie.kraskov_mi(x, y, H_X_given=H_X_given, k=bins)

	return MI

def calc_MI_samples_auto_bin(x, y):
	
	x = x.ravel()
	y = y.ravel()
	
	c_X = np.histogram(x, bins='auto')[0]
	c_Y = np.histogram(y, bins='auto')[0]
	bins = [len(c_X), len(c_Y)]
	c_XY = np.histogram2d(x, y, bins)[0]

	H_X = shan_entropy(c_X)
	H_Y = shan_entropy(c_Y)
	H_XY = shan_entropy(c_XY)

	MI = H_X + H_Y - H_XY
	
	return MI

from numpy.random import default_rng
def analytical_MI(noise_factor, dim, n_samples, random_seed):
	rng = default_rng(random_seed)

	# p(X)
	mu_x = np.atleast_1d(rng.random(dim))
	sig_x = np.atleast_2d(np.diag(rng.random(dim)))
	sig_x = sig_x @ sig_x.T

	# linear transformation matrix A
	A = np.atleast_2d(rng.random(dim))
	
	# noise distribution E
	mu_e = np.atleast_1d(np.zeros(1))
	sig_e = np.atleast_2d(np.ones(1)) * noise_factor

	# p(Y|X)
	# https://ssl2.cms.fu-berlin.de/ewi-psy/einrichtungen/arbeitsbereiche/computational_cogni_neurosc/PMFN/10-Gaussian-distributions.pdf eq. 10.20
	mu_y_x = A @ mu_x + mu_e
	sig_y_x = A @ sig_x @ A.T + sig_e

	# p(Y)
	# Bishop p.93 eq. 2.115
	mu_y = A@mu_x
	sig_y = sig_y_x + A@sig_x@A.T
	# p(X|Y)
	# Bishop p.93 eq. 2.116
	sig_x_y = np.linalg.inv(np.linalg.inv(sig_x) + A.T@np.linalg.inv(sig_y_x)@A)

	# p(X,Y)
	# sig_xy = np.block( [[sig_x,	sig_x@A.T], 
	# 					[A@sig_x,   sig_y_x + A@sig_x@A.T]] )
	sig_xy = np.block( [[sig_x,	sig_x@A.T], 
						[A@sig_x,   sig_y]] )

	H_x = entropy(sig_x)
	H_y = entropy(sig_y)
	H_xy = entropy(sig_xy)
	H_y_x = entropy(sig_y_x)
	H_x_y = entropy(sig_x_y)
	
	if sig_x.shape[0] > 1:

		H_x = 0
		for i in range(len(sig_x)):
			sig_xi = np.atleast_2d(sig_x[i][i])
			H_xi = entropy(sig_xi)
			H_x += H_xi

		C = sig_x @ A.T
		H_xy = 0
		for i in range(len(C)):
			sig_xiy = np.block( [[sig_x[i][i], C[i]],
								[C[i], sig_y_x + A@sig_x@A.T + sig_e]])
			H_xiy = entropy(sig_xiy)
			H_xy += H_xiy

	# equal expressions for the MI	
	I_0 = H_y - H_y_x
	I_1 = H_x - H_x_y
	I_2 = H_xy - H_x_y - H_y_x
	I_3 = H_x + H_y - H_xy
	
	assert np.all(np.linalg.eigvals(sig_x) > 0), 'sig_x not positive-definite'
	
	if noise_factor > 0:
		assert np.all(np.linalg.eigvals(sig_e) > 0), 'sig_e not positive-definite'

	x = rng.multivariate_normal(mu_x, sig_x, size=n_samples)
	e = rng.multivariate_normal(mu_e, sig_e, size=n_samples)

	# sample y values
	y = x@A.T + e

	return x, y, I_3, H_x

def compute_MI(x, y, I, H_x, bins):

	I_xy_sklearn_regression = []
	I_xy_sklearn = []
	I_xy_ksg = []
	I_xy_samples = []

	I_xy_samples_ab = 0

	legend = []

	for i, x_i in enumerate(x.T):

		x_tmp = x_i[:,np.newaxis] # N*d_x
		y_tmp = y # N*d_y -> N*1

		# for bin_i, bin in enumerate(bins):
		# 	 if i == 0:
		# 		 I_xy_sklearn_regression += [calc_MI_sklearn_regression(x_tmp, y_tmp, n_neighbors=bin)]
		# 	 else:
		# 		 I_xy_sklearn_regression[bin_i] += calc_MI_sklearn_regression(x_tmp, y_tmp, n_neighbors=bin)

		for bin_i, bin in enumerate(bins):
			if i == 0:
				I_xy_sklearn += [calc_MI_samples(x_tmp, y_tmp, bin)]
			else:
				I_xy_sklearn[bin_i] += calc_MI_samples(x_tmp, y_tmp, bin)

		for bin_i, bin in enumerate(bins):
			if i == 0:
				I_xy_samples += [calc_MI_prior(x_tmp, y_tmp, bin, H_X_given=H_x, n=x.shape[1])]
			else:
				I_xy_samples[bin_i] += calc_MI_prior(x_tmp, y_tmp, bin, H_X_given=H_x, n=x.shape[1])

		for bin_i, bin in enumerate(bins):
			if i == 0:
				I_xy_ksg += [calc_MI_KSG(x_tmp, y_tmp, bin)]
			else:
				I_xy_ksg[bin_i] += calc_MI_KSG(x_tmp, y_tmp, bin)

		I_xy_samples_ab += calc_MI_samples_auto_bin(x_tmp, y_tmp)

	# legend += ['$I_{regress}$']
	legend += ['$I_{binned}$']
	legend += ['$I_{prior}$']
	legend += ['$I_{KSG}$']

	legend += ['$I_{autobin}$']

	legend += ['$I$']

	# for i in range(len(bins)):
	# 	I_xy_sklearn[i] /= x.shape[1]
	# 	I_xy_sklearn_regression[i] /= x.shape[1]
	# 	I_xy_samples[i] /= x.shape[1]
	# 	I_xy_ksg[i] /= x.shape[1]
		
	# I_xy_samples_ab /= x.shape[1]

	colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:red']
	linestyles = ['-', ':', '--', '-.', 'solid', 'dashed', 'dashdot', 'dotted']
	assert len(linestyles) >= len(bins), 'define more linestyles'

	return [*I_xy_sklearn, *I_xy_samples, *I_xy_ksg, I_xy_samples_ab, I], legend, np.append(colors[:len(legend)-1], colors[-1]), linestyles
	# return [*I_xy_sklearn_regression, *I_xy_sklearn, *I_xy_samples, *I_xy_ksg, I_xy_samples_ab, I], legend, colors, linestyles

def plot_samples(dim=5, n_samples=[5, 10, 15], bins=[3, 4], noise_factor=2, n_runs=25, log_dir='', random_seed=42, override=False):
	mi_runs = []
	
	file_name = f'mi_dim_{dim}_noise_{noise_factor}_bins_{bins}'

	if os.path.isfile(os.path.join(log_dir, 'data', file_name+'.npy')) and not override: 
		mi_runs, legend, colors, linestyle = np.load(os.path.join(log_dir, 'data', file_name+'.npy'), allow_pickle=True)
		print('Open existing file')
	else:
		for n in n_samples:
			mi_trial = []
			for i in range(n_runs):
				x, y , I, H_x = analytical_MI(noise_factor, dim, n, random_seed=random_seed+i)
				mi, legend, colors, linestyle = compute_MI(x, y, I, H_x, bins)
				mi_trial += [mi]
			mi_runs += [mi_trial]
		np.save(open(os.path.join(log_dir, 'data', file_name+'.npy'), 'wb'), (mi_runs, legend, colors, linestyle))

	mi_runs_mean = np.array(mi_runs).mean(axis=1).T

	mi_runs_std = np.array(mi_runs).std(axis=1).T

	fig, ax = plt.subplots()

	x = n_samples
	
	for i, (mi_run_mean, mi_run_std) in enumerate(zip(mi_runs_mean[:-1], mi_runs_std[:-1])):
		ax.plot(x, mi_run_mean, color=colors[:-1][i//(len(bins))], linestyle=linestyle[i%(len(bins))])
		ci = mi_run_std*2
		# ax.fill_between(x, mi_run_mean-ci, mi_run_mean+ci, alpha=.2, color=colors[:-1][i//(len(bins))])

	ax.plot(x, mi_runs_mean[-1], color=colors[-1], linestyle='-')

	legend_elements = []
	for color, leg in zip(colors, legend):
		legend_elements += [Line2D([0], [0], color=color, lw=1, label=leg)]
	
	if len(colors) % 2 != 0:
		legend_elements += [Line2D([0], [0], color='white', lw=1, label='')]

	for i, bin in enumerate(bins):
		legend_elements += [Line2D([0], [0], color='black', lw=1, label=f'bins/$k$={bin}', linestyle=linestyle[i])]


	ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=5)
	plt.subplots_adjust(bottom=0.25)

	ax.set_xlabel('samples')
	ax.set_ylabel('mutual information')
	# plt.title(f'dimensionality: {dim}, var noise: {noise_factor}')

	plt.savefig(os.path.join(log_dir, 'imgs', file_name+'.pdf'))
	plt.savefig(os.path.join(log_dir, 'imgs', file_name+'.png'))

if __name__ == '__main__':

	log_dir = 'logs_mi'
	img_dir = os.path.join(log_dir, 'imgs')
	data_dir = os.path.join(log_dir, 'data')
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(img_dir, exist_ok=True)
	os.makedirs(data_dir, exist_ok=True)

	plot_samples(dim=1, n_samples=np.arange(15, 100, 1), bins=[1, 3], noise_factor=1., n_runs=25, log_dir=log_dir, random_seed=42, override=False)

	# plot_samples(dim=140, n_samples=np.arange(20, 200, 5), bins=[3, 4, 5, 10], noise_factor=1, n_runs=10, log_dir=log_dir)

	# plot_samples(dim=140, n_samples=np.arange(20, 200, 5), bins=[19], noise_factor=1, n_runs=10, log_dir=log_dir)
	