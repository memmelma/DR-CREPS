#Written by Weihao Gao from UIUC
#https://github.com/wgao9/knnie
import scipy.spatial as ss
import scipy.stats as sst
from scipy.special import digamma,gamma
from sklearn.neighbors import KernelDensity
from math import log,pi,exp
import numpy.random as nr
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from cvxopt import matrix,solvers


#Usage Functions
def kraskov_mi(x,y, H_X_given=None, k=5):
	'''
		Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
		Using KSG mutual information estimator

		Input: x: 2D list of size N*d_x
		y: 2D list of size N*d_y
		k: k-nearest neighbor parameter

		Output: one number of I(X;Y)
	'''

	assert len(x)==len(y), "Lists should have same length"
	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	dx = len(x[0])	
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)

	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

	knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
	ans_xy = -digamma(k) + digamma(N) + (dx+dy)*log(2)#2*log(N-1) - digamma(N) #+ vd(dx) + vd(dy) - vd(dx+dy)
	ans_x = digamma(N) + dx*log(2)
	ans_y = digamma(N) + dy*log(2)
	for i in range(N):
		ans_xy += (dx+dy)*log(knn_dis[i])/N
		ans_x += -digamma(len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dx*log(knn_dis[i])/N
		ans_y += -digamma(len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dy*log(knn_dis[i])/N
	
	if H_X_given is not None:
		return H_X_given+ans_y-ans_xy
	else:
		return ans_x+ans_y-ans_xy

from sklearn.neighbors import NearestNeighbors, KDTree

def kraskov_mi_sklearn(x,y,n_neighbors=5):
	n_samples = x.size

	x = x.reshape((-1, 1))
	y = y.reshape((-1, 1))
	xy = np.hstack((x, y))

	# Here we rely on NearestNeighbors to select the fastest algorithm.
	nn = NearestNeighbors(metric='chebyshev', n_neighbors=n_neighbors)

	nn.fit(xy)
	radius = nn.kneighbors()[0]
	radius = np.nextafter(radius[:, -1], 0)

	# KDTree is explicitly fit to allow for the querying of number of
	# neighbors within a specified radius
	kd = KDTree(x, metric='chebyshev')
	nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
	nx = np.array(nx) - 1.0

	kd = KDTree(y, metric='chebyshev')
	ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
	ny = np.array(ny) - 1.0

	mi = (digamma(n_samples) + digamma(n_neighbors) - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)))

	return max(0, mi)

from scipy.sparse import issparse
from sklearn.utils.validation import check_array, check_X_y
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from sklearn.utils.fixes import _astype_copy_false

# sklearn
def _estimate_mi(X, y, discrete_features='auto', discrete_target=False,
				 n_neighbors=3, copy=True, random_state=None):

	X, y = check_X_y(X, y, accept_sparse='csc', y_numeric=not discrete_target)
	n_samples, n_features = X.shape

	if isinstance(discrete_features, (str, bool)):
		if isinstance(discrete_features, str):
			if discrete_features == 'auto':
				discrete_features = issparse(X)
			else:
				raise ValueError("Invalid string value for discrete_features.")
		discrete_mask = np.empty(n_features, dtype=bool)
		discrete_mask.fill(discrete_features)
	else:
		discrete_features = check_array(discrete_features, ensure_2d=False)
		if discrete_features.dtype != 'bool':
			discrete_mask = np.zeros(n_features, dtype=bool)
			discrete_mask[discrete_features] = True
		else:
			discrete_mask = discrete_features

	continuous_mask = ~discrete_mask
	if np.any(continuous_mask) and issparse(X):
		raise ValueError("Sparse matrix `X` can't have continuous features.")

	rng = check_random_state(random_state)
	if np.any(continuous_mask):
		if copy:
			X = X.copy()

		if not discrete_target:
			X[:, continuous_mask] = scale(X[:, continuous_mask],
										  with_mean=False, copy=False)

		# Add small noise to continuous features as advised in Kraskov et. al.
		X = X.astype(float, **_astype_copy_false(X))
		means = np.maximum(1, np.mean(np.abs(X[:, continuous_mask]), axis=0))

		X[:, continuous_mask] += 1e-10 * means * rng.randn(
				n_samples, np.sum(continuous_mask))

	if not discrete_target:
		y = scale(y, with_mean=False)
		y += 1e-10 * np.maximum(1, np.mean(np.abs(y))) * rng.randn(n_samples)

	mi = [_compute_mi(x, y, discrete_feature, discrete_target, n_neighbors) for
		  x, discrete_feature in zip(_iterate_columns(X), discrete_mask)]

	return np.array(mi)

# sklearn
def _compute_mi(x, y, x_discrete, y_discrete, n_neighbors=3):
	"""Compute mutual information between two variables.
	This is a simple wrapper which selects a proper function to call based on
	whether `x` and `y` are discrete or not.
	"""
	if x_discrete and y_discrete:
		print('0')
		exit()
		# return mutual_info_score(x, y)
	elif x_discrete and not y_discrete:
		# return _compute_mi_cd(y, x, n_neighbors)
		print('1')
		exit()
	elif not x_discrete and y_discrete:
		# return _compute_mi_cd(x, y, n_neighbors)
		print('2')
		exit()
	else:
		return kraskov_mi_sklearn(x, y, n_neighbors)

# sklearn
def _iterate_columns(X, columns=None):
	"""Iterate over columns of a matrix.
	Parameters
	----------
	X : ndarray or csc_matrix, shape (n_samples, n_features)
		Matrix over which to iterate.
	columns : iterable or None, default=None
		Indices of columns to iterate over. If None, iterate over all columns.
	Yields
	------
	x : ndarray, shape (n_samples,)
		Columns of `X` in dense format.
	"""
	if columns is None:
		columns = range(X.shape[1])

	if issparse(X):
		for i in columns:
			x = np.zeros(X.shape[0])
			start_ptr, end_ptr = X.indptr[i], X.indptr[i + 1]
			x[X.indices[start_ptr:end_ptr]] = X.data[start_ptr:end_ptr]
			yield x
	else:
		for i in columns:
			yield X[:, i]

def revised_mi(x,y,k=5,q=float('inf')):
	'''
		Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
		Using *REVISED* KSG mutual information estimator (see arxiv.org/abs/1604.03006)

		Input: x: 2D list of size N*d_x
		y: 2D list of size N*d_y
		k: k-nearest neighbor parameter
		q: l_q norm used to decide k-nearest distance

		Output: one number of I(X;Y)
	'''

	assert len(x)==len(y), "Lists should have same length"
	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	dx = len(x[0])	
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)

	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

	knn_dis = [tree_xy.query(point,k+1,p=q)[0][k] for point in data]
	ans_xy = -digamma(k) + log(N) + vd(dx+dy,q)
	ans_x = log(N) + vd(dx,q)
	ans_y = log(N) + vd(dy,q)
	for i in range(N):
		ans_xy += (dx+dy)*log(knn_dis[i])/N
		ans_x += -log(len(tree_x.query_ball_point(x[i],knn_dis[i]+1e-15,p=q))-1)/N+dx*log(knn_dis[i])/N
		ans_y += -log(len(tree_y.query_ball_point(y[i],knn_dis[i]+1e-15,p=q))-1)/N+dy*log(knn_dis[i])/N		
	return ans_x+ans_y-ans_xy


def kraskov_multi_mi(x,y,z,k=5):
	'''
		Estimate the multivariate mutual information I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y,Z)
		of X, Y and Z from samples {x_i, y_i, z_i}_{i=1}^N
		Using KSG mutual information estimator

		Input: x: 2D list of size N*d_x
		y: 2D list of size N*d_y
		z: 2D list of size N*d_z
		k: k-nearest neighbor parameter

		Output: one number of I(X;Y;Z)
	'''

	assert len(x)==len(y), "Lists should have same length"
	assert len(x)==len(z), "Lists should have same length"
	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	dx = len(x[0])	
	dy = len(y[0])
	dz = len(z[0])
	data = np.concatenate((x,y,z),axis=1)

	tree_xyz = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)
	tree_z = ss.cKDTree(z)

	knn_dis = [tree_xyz.query(point,k+1,p=float('inf'))[0][k] for point in data]
	ans_xyz = -digamma(k) + digamma(N) + (dx+dy+dz)*log(2)#2*log(N-1) - digamma(N) #+ vd(dx) + vd(dy) - vd(dx+dy)
	ans_x = digamma(N) + dx*log(2)
	ans_y = digamma(N) + dy*log(2)
	ans_z = digamma(N) + dz*log(2)
	for i in range(N):
		ans_xyz += (dx+dy+dz)*log(knn_dis[i])/N
		ans_x += -digamma(len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dx*log(knn_dis[i])/N
		ans_y += -digamma(len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dy*log(knn_dis[i])/N
		ans_z += -digamma(len(tree_z.query_ball_point(z[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dz*log(knn_dis[i])/N

	return ans_x+ans_y+ans_z-ans_xyz


def revised_multi_mi(x,y,z,k=5,q=float('inf')):

	'''
		Estimate the multivariate mutual information I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y,Z)
		of X, Y and Z from samples {x_i, y_i, z_i}_{i=1}^N
		Using *REVISED* KSG mutual information estimator (see arxiv.org/abs/1604.03006)

		Input: x: 2D list of size N*d_x
		y: 2D list of size N*d_y
		z: 2D list of size N*d_z
		k: k-nearest neighbor parameter
		q: l_q norm used to decide k-nearest neighbor distance

		Output: one number of I(X;Y;Z)
	'''
	assert len(x)==len(y), "Lists should have same length"
	assert len(x)==len(z), "Lists should have same length"
	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	dx = len(x[0])	
	dy = len(y[0])
	dz = len(z[0])
	data = np.concatenate((x,y,z),axis=1)

	tree_xyz = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)
	tree_z = ss.cKDTree(z)

	knn_dis = [tree_xyz.query(point,k+1,p=q)[0][k] for point in data]
	ans_xyz = -digamma(k) + log(N) + vd(dx+dy+dz,q)
	ans_x = log(N) + vd(dx,q)
	ans_y = log(N) + vd(dy,q)
	ans_z = log(N) + vd(dz,q)
	for i in range(N):
		ans_xyz += (dx+dy+dz)*log(knn_dis[i])/N
		ans_x += -log(len(tree_x.query_ball_point(x[i],knn_dis[i]+1e-15,p=q))-1)/N+dx*log(knn_dis[i])/N
		ans_y += -log(len(tree_y.query_ball_point(y[i],knn_dis[i]+1e-15,p=q))-1)/N+dy*log(knn_dis[i])/N		
		ans_z += -log(len(tree_z.query_ball_point(z[i],knn_dis[i]+1e-15,p=q))-1)/N+dz*log(knn_dis[i])/N		
	return ans_x+ans_y+ans_z-ans_xyz


#Auxilary functions
def vd(d,q):
	# Compute the volume of unit l_q ball in d dimensional space
	if (q==float('inf')):
		return d*log(2)
	return d*log(2*gamma(1+1.0/q)) - log(gamma(1+d*1.0/q))

def entropy(x,k=5,q=float('inf')):
	# Estimator of (differential entropy) of X 
	# Using k-nearest neighbor methods 
	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	d = len(x[0]) 	
	thre = 3*(log(N)**2/N)**(1/d)
	tree = ss.cKDTree(x)
	knn_dis = [tree.query(point,k+1,p=q)[0][k] for point in x]
	truncated_knn_dis = [knn_dis[s] for s in range(N) if knn_dis[s] < thre]
	ans = -digamma(k) + digamma(N) + vd(d,q)
	return ans + d*np.mean(map(log,knn_dis))

def kde_entropy(x):
	# Estimator of (differential entropy) of X 
	# Using resubstitution of KDE
	N = len(x)
	d = len(x[0])
	local_est = np.zeros(N)
	for i in range(N):
		kernel = sst.gaussian_kde(x.transpose())
		local_est[i] = kernel.evaluate(x[i].transpose())
	return -np.mean(map(log,local_est))










