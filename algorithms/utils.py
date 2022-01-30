import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression


def compute_corr(theta_rot, Jep, k, C, estimator):

    if C == 'Random':
        corr = None
    elif C == 'MI':
        corr = compute_mi(theta_rot, Jep, estimator=estimator)
    elif C == 'PCC':
        corr = compute_pcc(theta_rot, Jep)

    # select important parameters
    if corr is not None:
        corr = corr / np.max(corr)
        top_k_corr = corr.argsort()[-int(k):][::-1]
    else:
        rng = np.random.default_rng()
        top_k_corr = rng.choice(list(range(0, theta_rot.shape[1], 1)), size=k, replace=False)
 
    return top_k_corr, corr

def compute_mi(theta, Jep, estimator='regression'):
    if estimator == 'score':
        MI = []
        for theta_i in theta.T:
            c_xy = np.histogram2d(theta_i, Jep, bins=4)[0]
            MI += [mutual_info_score(None, None, contingency=c_xy)]
        MI = np.array(MI)
    elif estimator == 'hist':
        MI = []
        for theta_i in theta.T:
            MI += [mutual_info_histogram_entropy(theta_i, Jep, bins=4)]
        MI = np.array(MI)
    elif estimator == 'regression':
        MI = mutual_info_regression(theta, Jep, discrete_features=False, n_neighbors=4, random_state=42)
    return MI

def compute_pcc(theta, Jep):
    PCC = []
    for i in range(theta.shape[1]):
        PCC += [pearsonr(theta[:,i], Jep)[0]]
    return np.abs(PCC)

def mutual_info_histogram_entropy(x, y, bins=4):
    # density estimation via histograms
    c_XY = np.histogram2d(x, y, bins)[0]
    c_X = np.histogram(x, bins)[0]
    c_Y = np.histogram(y, bins)[0]
    # compute Shannon entropy
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)
    # compute MI
    return H_X + H_Y - H_XY

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log(c_normalized))  
    return H