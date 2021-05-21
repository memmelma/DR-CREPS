import os
import numpy as np
import matplotlib.pyplot as plt

# closed form entropy as in REPS compendium 
def entropy(sig):
    n = sig.shape[0]
    return 0.5*np.linalg.slogdet(sig)[1] + (n*np.log(2*np.pi))/2 + n/2


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


from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
def calc_MI_sklearn(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def calc_MI(X,Y,bins,H_X_given):
    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)
    # print('H_X, H_X_given', H_X, H_X_given)
    MI = H_X + H_Y - H_XY
    return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

def get_mi(noise_factor=1, dim=5, n_samples=1000, bins=[3,5]):

    # p(X)
    mu_x = np.atleast_1d(np.random.rand(dim))
    sig_x = np.atleast_2d(np.random.rand(dim,dim))
    sig_x = sig_x @ sig_x.T
    # linear transformation matrix A
    A = np.atleast_2d(np.random.rand(dim,dim))
    
    # noise distribution E
    mu_e = np.atleast_1d(np.random.rand(dim))
    sig_e = np.atleast_2d(np.random.rand(dim,dim))
    sig_e = sig_e @ sig_e.T * noise_factor

    # p(Y|X)
    # https://ssl2.cms.fu-berlin.de/ewi-psy/einrichtungen/arbeitsbereiche/computational_cogni_neurosc/PMFN/10-Gaussian-distributions.pdf eq. 10.20
    mu_y_x = A @ mu_x + mu_e
    sig_y_x = A @ sig_x @ A.T + sig_e

    # p(Y)
    # Bishop p.93 eq. 2.115
    mu_y = A@mu_x + mu_e
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

    # equal expressions for the MI    
    I = H_y - H_y_x
    # print('H_y - H_y_x', I)
    I = H_x - H_x_y
    # print('H_x - H_x_y', I)
    I = H_xy - H_x_y - H_y_x
    # print('H_xy - H_x_y - H_y_x', I)
    I = H_x + H_y - H_xy
    # print('H_x + H_y - H_xy', I)

    assert np.all(np.linalg.eigvals(sig_x) > 0), 'sig_x not positive-definite'
    
    if noise_factor > 0:
        assert np.all(np.linalg.eigvals(sig_e) > 0), 'sig_e not positive-definite'

    x = np.random.multivariate_normal(mu_x, sig_x, size=n_samples).T
    e = np.random.multivariate_normal(mu_e, sig_e, size=n_samples).T

    # sample y values
    y = A@x + e

    I_xy_sklearn_regression = 0
    I_xy_sklearn = []
    I_xy_scipy = 0
    I_xy_samples = 0

    legend = []

    for i in range(x.shape[0]):
        x_tmp = np.atleast_2d(x[i]).T
        I_xy_sklearn_regression += calc_MI_sklearn_regression(x_tmp, y[i].squeeze(), bins)
        if i == 0:
            legend += ['I_xy_sklearn_regression']
        for bin_i, bin in enumerate(bins):
            if i == 0:
                I_xy_sklearn += [calc_MI_sklearn(x_tmp.squeeze(), y[i].squeeze(), bin)]
                legend += [f'I_xy_sklearn_bins_{bin}']
            else:
                I_xy_sklearn[bin_i] += calc_MI_sklearn(x_tmp.squeeze(), y[i].squeeze(), bin)

        # I_xy_scipy += calc_MI_scipy(x_tmp.squeeze(), y[i].squeeze(), bins[0])
        # I_xy_samples += calc_MI(x_tmp.squeeze(), y[i].squeeze(), bins[0], H_X_given=H_x)
        # if i == 0:
        #     legend += [f'I_xy_scipy_bins_{bins[0]}']
        #     legend += [f'I_xy_samples_bins_{bins[0]}']
    legend += ['I']

    return np.array([I_xy_sklearn_regression, *I_xy_sklearn, I_xy_scipy, I_xy_samples, I]), legend


def plot_standard(dim = 5, n_samples = 500, bins = [3], noise_factors = np.arange(1, 10, 0.1), n_runs = 25):
    mi_runs = []
    
    for noise in noise_factors:
        np.random.seed(42)
        mi_trial = []
        for i in range(n_runs):
            mi, legend = get_mi(noise_factor=noise, dim=dim, n_samples=n_samples, bins=bins)
            mi_trial += [mi]
        mi_runs += [mi_trial]

    mi_runs_mean = np.array(mi_runs).mean(axis=1).T
    mi_runs_std = np.array(mi_runs).std(axis=1).T

    fig, ax = plt.subplots()

    x = noise_factors # np.arange(0, mi_runs_mean.shape[1], 1)

    for mi_run_mean, mi_run_std in zip(mi_runs_mean, mi_runs_std):
        ax.plot(x, mi_run_mean)
        ci = mi_run_std*2
        ax.fill_between(x, mi_run_mean-ci, mi_run_mean+ci, alpha=.3)

    # ax.legend(['I_xy_sklearn_regression', 'I_xy_sklearn', 'I_xy_scipy', 'I'])
    ax.legend(legend)
    ax.set_xlabel('noise')
    ax.set_ylabel('average mutual information')
    plt.title(f'dimensionality: {dim}, samples: {n_samples}, bins: {bins}')
    plt.savefig(os.path.join(img_dir, f'mi_dim_{dim}_samples_{n_samples}_bins_{bins}.png'))

def plot_samples(dim = 5, n_samples = [500], bins = [3], noise_factor = 2, n_runs = 25):
    mi_runs = []
    
    for n in n_samples:
        np.random.seed(42)
        mi_trial = []
        for i in range(n_runs):
            mi, legend = get_mi(noise_factor=noise_factor, dim=dim, n_samples=n, bins=bins)
            mi_trial += [mi]
        mi_runs += [mi_trial]

    mi_runs_mean = np.array(mi_runs).mean(axis=1).T
    mi_runs_std = np.array(mi_runs).std(axis=1).T

    fig, ax = plt.subplots()

    x = n_samples # np.arange(0, mi_runs_mean.shape[1], 1)

    for mi_run_mean, mi_run_std, label in zip(mi_runs_mean, mi_runs_std, legend):
        ax.plot(x, mi_run_mean, label=label)
        ci = mi_run_std*2
        ax.fill_between(x, mi_run_mean-ci, mi_run_mean+ci, alpha=.3)

    # ax.legend(['I_xy_sklearn_regression', 'I_xy_sklearn', 'I_xy_scipy', 'I'])
    ax.legend()
    ax.set_xlabel('samples')
    ax.set_ylabel('average mutual information')
    plt.title(f'dimensionality: {dim}, samples: {min(n_samples), max(n_samples)}, bins: {bins}')
    plt.savefig(os.path.join(img_dir, f'mi_dim_{dim}_noise_{noise_factor}_bins_{bins}.png'))

if __name__ == '__main__':

    img_dir = 'mi_toy'
    os.makedirs(img_dir, exist_ok=True)

    n_runs = 10

    # for dim in [5, 10]:
    #     for n_samples in [25, 100]:
    #         plot_standard(dim = dim, n_samples = n_samples, bins = [2, 3, 5], noise_factors = np.arange(1, 50, 1), n_runs=n_runs)
            
    for dim in [1, 5, 10]:
        for noise_factor in [0, 5, 10]:
            plot_samples(dim = dim, n_samples = np.arange(5, 250, 5), bins = [2, 3, 5, 10], noise_factor = noise_factor, n_runs=n_runs)
    