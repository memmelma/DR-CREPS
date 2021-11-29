from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

    local = False
    test = False

    env_seed = 0
    lqr_dim = 10
    eff = 3

    experiment_name = f'lqr_ppo_trpo_reinforce_search_prepro'

    launcher = Launcher(experiment_name,
                        'el_lqr_rebuttal',
                        25,
                        memory=500,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=False,
                        conda_env='iprl')
	
    ### LQR DIAGONAL COVARIANCE ######################################################################################################################################################

    # launcher.add_default_params(lqr_dim=lqr_dim, n_ineff=lqr_dim-eff, env_seed=env_seed, 
    #                             n_epochs=100, ep_per_fit=25, fit_per_epoch=1,
    #                             sigma_init=3e-1,
    #                             bins=4, mi_type='regression')

    # # Diagonal Gaussian Distribution
    # distribution = 'diag'
    
    # # REPS & RWR
    # for eps in np.arange(0.1, 1.5, 0.1):
    #     eps = np.round(eps, 1)
    #     launcher.add_experiment(alg='REPS', distribution=distribution, eps=eps)
    #     launcher.add_experiment(alg='RWR', distribution=distribution, eps=eps)

    # # Constrained REPS
    # for eps in np.arange(1.9, 3.5, 0.2):
    #     eps = np.round(eps, 1)
    #     for kappa in np.arange(1., 10., 1.):
    #         kappa = np.round(kappa, 1)
    #         launcher.add_experiment(alg='ConstrainedREPS', distribution=distribution, eps=eps, kappa=kappa)

    # # Best eps from REPS
    # eps = 0.4

    # # REPS MI
    # for k in np.arange(5, 100, 5):
    #     k = int(k)
    #     for gama in np.arange(0.1, 1, 0.1):
    #         gama = np.round(gama, 1)
    #         launcher.add_experiment(alg='REPS_MI', distribution=distribution, eps=eps, k=k, sample_type='percentage', gamma=gama, method='MI')
    
    # # REPS MI Oracle
    # for eps in np.arange(0.1, 1.9, 0.2):
    #     eps = np.round(eps, 1)
    #     for gama in np.arange(0.1, 1, 0.1):
    #         gama = np.round(gama, 1)
    #         launcher.add_experiment(alg='REPS_MI_ORACLE', distribution=distribution, eps=eps, sample_type='percentage', gamma=gama, method='MI')
    
    # # Best eps and kappa from Constrained REPS
    # eps = 2.5
    # kappa = 6.0

    # # Constrained REPS MI
    # for k in np.arange(5, 100, 5):
    #     k = int(k)
    #     for gama in np.arange(0.1, 1, 0.1):
    #         gama = np.round(gama, 1)
    #         launcher.add_experiment(alg='ConstrainedREPSMI', distribution=distribution, eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=gama, method='MI')
    
    # # Constrained REPS MI Oracle
    # for eps in np.arange(1.9, 3.5, 0.2):
    #     eps = np.round(eps, 1)
    #     for kappa in np.arange(1., 10., 1.):
    #         kappa = np.round(kappa, 1)
    #         for gama in np.arange(0.1, 1, 0.1):
    #             gama = np.round(gama, 1)
    #             launcher.add_experiment(alg='ConstrainedREPSMIOracle', distribution=distribution, eps=eps, kappa=kappa, sample_type='percentage', gamma=gama, method='MI')
    
    # # Best eps from RWR
    # eps = 0.2

    # # RWR MI
    # for k in np.arange(10, 100, 10):
    #     k = int(k)
    #     for gama in np.arange(0.1, 1, 0.1):
    #         gama = np.round(gama, 1)
    #         launcher.add_experiment(alg='RWR_MI', distribution=distribution, eps=eps, k=k, sample_type='percentage', gamma=gama, method='MI')
    
    # launcher.add_experiment(alg='RWR_MI', distribution=distribution, eps=eps, sample_type='PRO', method='Pearson')
    # launcher.add_experiment(alg='RWR_MI', distribution=distribution, eps=eps, sample_type='PRO', method='MI')

    ### LQR FULL COVARIANCE ######################################################################################################################################################

    # launcher.add_default_params(lqr_dim=lqr_dim, n_ineff=lqr_dim-eff, env_seed=env_seed, 
    #                             fit_per_epoch=1,
    #                             sigma_init=3e-1,
    #                             bins=4, mi_type='regression',
    #                             nn_policy=0)

    # n_samples = 5000

    # # # Cholesky Gaussian Distribution
    # distribution = 'cholesky'

    # ep_per_fit = 250
    # n_epochs = n_samples // ep_per_fit

    # # REPS & RWR
    # for eps in np.arange(0.1, 2.5, 0.1):
    #     eps = np.round(eps, 1)
    #     launcher.add_experiment(alg='REPS', distribution=distribution, eps=eps, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #     launcher.add_experiment(alg='RWR', distribution=distribution, eps=eps, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    # # Constrained REPS
    # for eps in np.arange(1.9, 6.0, 0.2):
    #     eps = np.round(eps, 1)
    #     for kappa in np.arange(1., 20., 1.):
    #         kappa = np.round(kappa, 1)
    #         launcher.add_experiment(alg='ConstrainedREPS', distribution=distribution, eps=eps, kappa=kappa, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    # # # MORE
    # for eps in np.arange(1.9, 6.0, 0.2):
    #     eps = np.round(eps, 1)
    #     for kappa in np.arange(1., 20., 1.):
    #         kappa = np.round(kappa, 1)
    #         launcher.add_experiment(alg='MORE', distribution=distribution, eps=eps, kappa=kappa, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    # Cholesky Gaussian Distribution for MI based algorithms
    # Find appropriate # of samples for given k
    # distribution = 'mi'

    # Best eps from REPS
    # for eps in [0.3, 0.5, 0.7]:
    # for eps in [0.1, 0.2, 0.3, 0.4]:
    #     for k in [20, 30, 40, 50, 60]:
    #         # for ep_per_fit in np.arange(k, 100, 10):
    #         #     n_epochs = n_samples // ep_per_fit
    #         ep_per_fit = int(k * 2.5)
    #         n_epochs = n_samples // ep_per_fit
    #         launcher.add_experiment(alg='REPS_MI_full', distribution=distribution, eps=eps, k=k, sample_type=None, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #         launcher.add_experiment(alg='REPS_MI_full', distribution=distribution, eps=eps, k=k, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit, sample_type='percentage', gamma=0.9)
    
    # launcher.add_experiment(alg='REPS_MI_full', distribution=distribution, eps=0.8, k=55, sample_type=None, method='MI', n_epochs=125, ep_per_fit=60)
    # launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution=distribution, eps=3.1, kappa=9., k=55, sample_type=None, method='MI', n_epochs=125, ep_per_fit=60)


    # n_samples = 5000

    # # Cholesky Gaussian Distribution for MI based algorithms
    # distribution = 'mi'

    # # Best eps from REPS
    # eps = 0.7
    
    # # REPS MI
    # # for k in np.arange(5, 100, 5):
    # for k in np.arange(5, 60, 10):
    #     k = int(k)
    #     # n_samples required for full covariance / n_parameters = 2.5
    #     # ep_per_fit = int(k * 3)
    #     for ep_per_fit in [k*3, k*4]:
    #         n_epochs = n_samples // ep_per_fit
        
    #         for gama in np.arange(0.1,1.0,0.1):
    #             gama = np.round(gama,1)
    #             launcher.add_experiment(alg='REPS_MI_full', distribution=distribution, eps=eps, k=k, sample_type='percentage', gamma=gama, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #     # launcher.add_experiment(alg='REPS_MI_full', distribution=distribution, eps=eps, k=k, sample_type=None, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    
    # for eps in [1.1, 1.3]:
    #     for k in range(10, 60, 20):
    #         for ep_per_fit in np.arange(50, 310, 50):
    #             n_epochs = n_samples // ep_per_fit
    #             launcher.add_experiment(alg='REPS_MI_full', distribution=distribution, eps=eps, k=k, sample_type='percentage', gamma=0.1, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #             launcher.add_experiment(alg='REPS_MI_full', distribution=distribution, eps=eps, k=k, sample_type='percentage', gamma=0.5, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #             launcher.add_experiment(alg='REPS_MI_full', distribution=distribution, eps=eps, k=k, sample_type='percentage', gamma=0.9, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    # # Best eps and kappa from Constrained REPS
    # eps = 4.7
    # kappa = 17.
    
    # # Constrained REPS MI
    # for k in np.arange(5, 100, 5):
    #     k = int(k)
    #     ep_per_fit = 125 # k
    #     n_epochs = n_samples // ep_per_fit
        
    #     for gama in np.arange(0.1,1.0,0.1):
    #         gama = np.round(gama,1)
    #         launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution=distribution, eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=gama, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #     launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution=distribution, eps=eps, kappa=kappa, k=k, sample_type=None, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
        


    #     eps_adapt = np.round(eps*k/100,1)
    #     kappa_adapt = int(kappa**k/100)
    #     for gama in np.arange(0.1,1.0,0.1):
    #         gama = np.round(gama,1)
    #         launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution=distribution, eps_adapt=eps, kappa=kappa_adapt, k=k, sample_type='percentage', gamma=gama, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #     launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution=distribution, eps_adapt=eps, kappa=kappa_adapt, k=k, sample_type=None, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)


    # eps = 4.
    # kappa = 10.
    # ep_per_fit = 90
    # n_epochs = n_samples // ep_per_fit
    
    # launcher.add_experiment(alg='ConstrainedREPS', distribution='cholesky', eps=eps, kappa=kappa, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    # launcher.add_experiment(alg='ConstrainedREPS', distribution='diag', eps=eps, kappa=kappa, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    
    
    # for method in ['Pearson', 'MI']:
    #     launcher.add_experiment(alg='ConstrainedREPSMI', distribution='diag', eps=eps, kappa=kappa, sample_type='importance', method=method, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #     launcher.add_experiment(alg='ConstrainedREPSMI', distribution='diag', eps=eps, kappa=kappa, sample_type='PRO', method=method, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #     for k in [10, 30, 50, 70]:
    #         for gama in [0.1, 0.5, 0.9]:
    #             launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=gama, method=method, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #             launcher.add_experiment(alg='ConstrainedREPSMI', distribution='diag', eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=gama, method=method, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    
    # eps = 0.7
    # ep_per_fit = 90
    # n_epochs = n_samples // ep_per_fit
    
    # launcher.add_experiment(alg='REPS', distribution='cholesky', eps=eps, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    # launcher.add_experiment(alg='REPS', distribution='diag', eps=eps, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    
    
    # for method in ['Pearson', 'MI']:
    #     launcher.add_experiment(alg='REPS_MI', distribution='diag', eps=eps, sample_type='importance', method=method, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #     launcher.add_experiment(alg='REPS_MI', distribution='diag', eps=eps, sample_type='PRO', method=method, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #     for k in [10, 30, 50, 70]:
    #         for gama in [0.1, 0.5, 0.9]:
    #             launcher.add_experiment(alg='REPS_MI_full', distribution='mi', eps=eps, k=k, sample_type='percentage', gamma=gama, method=method, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #             launcher.add_experiment(alg='REPS_MI', distribution='diag', eps=eps, k=k, sample_type='percentage', gamma=gama, method=method, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    # launcher.add_experiment(alg='REPS', distribution='cholesky', eps=eps, sample_type=None, n_epochs=n_samples//250, ep_per_fit=250)

    # launcher.add_default_params(lqr_dim=lqr_dim, n_ineff=lqr_dim-eff, env_seed=env_seed, 
    #                             fit_per_epoch=1,
    #                             sigma_init=3e-1,
    #                             # bins=4, mi_type='regression',
    #                             nn_policy=0) # bins=8

    # n_samples = 7500

    # for gama in [0.1, 0.3, 0.5]:
    #     for k in [30, 50]:
    #         eps = 0.7
    #         ep_per_fit = 100
    #         n_epochs = n_samples // ep_per_fit
    #         for method in ['MI', 'Pearson', 'Random']:
    #             launcher.add_experiment(alg='REPS_MI', distribution='diag', eps=eps, k=k, sample_type='percentage', method=method, gamma=gama, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    # launcher.add_experiment(alg='REPS', distribution='diag', sample_type=None, eps=eps, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    # n_samples = 7500
    
    # kappa = 15.
    # ep_per_fit = 250
    # n_epochs = n_samples // ep_per_fit
    # eps = 4.9 # best

    # # launcher.add_experiment(alg='ConstrainedREPS', distribution='cholesky', sample_type=None, eps=eps, kappa=kappa, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    # for gama in [0.1, 0.3, 0.5]:
    #     # for k in [10, 30, 50]:
    #     for k in [30, 50, 100]:
    #         for method in ['MI', 'Pearson', 'Random']:
    #             launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=eps, kappa=kappa, k=k, sample_type='percentage', method=method, gamma=gama, n_epochs=n_epochs, ep_per_fit=ep_per_fit)


#     n_samples = 4000
#     eps = 4.5
#     kappa = 15.
#     k = 10 # 30
#     # for ep_per_fit in [50]:
#     ep_per_fit = 50
# # for ep_per_fit in [250]:
#     n_epochs = n_samples // ep_per_fit
#     # for mi_type in ['regression']:
#         # for bins in range(3, 25, 10):
#     # launcher.add_experiment(alg='ConstrainedREPSMI', distribution='diag', bins=10, mi_type='regression', eps=eps, kappa=kappa, k=30, sample_type='percentage', method='MI', gamma=0.1, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
#     # launcher.add_experiment(alg='ConstrainedREPSMI', distribution='diag', bins=10, mi_type='regression', eps=eps, kappa=kappa, k=30, sample_type='percentage', method='MI_ALL', gamma=0.1, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
#     # launcher.add_experiment(alg='ConstrainedREPSMI', distribution='diag', bins=10, mi_type='regression', eps=eps, kappa=kappa, k=30, sample_type='percentage', method='Pearson', gamma=0.1, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
#     for ep_per_fit in [50, 100]:
   
#         n_epochs = n_samples // ep_per_fit
#         launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', bins=10, mi_type='regression', eps=eps, kappa=kappa, k=k, sample_type='percentage', method='MI', gamma=0.1, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
#         launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', bins=10, mi_type='regression', eps=eps, kappa=kappa, k=k, sample_type='percentage', method='MI_ALL', gamma=0.1, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
#         launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', bins=10, mi_type='regression', eps=eps, kappa=kappa, k=k, sample_type='percentage', method='Pearson', gamma=0.1, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

#     # launcher.add_experiment(alg='REPS_MI_ORACLE', distribution='diag', eps=eps, k=k, sample_type='percentage', gamma=gama, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    

    # n_samples = 5000

    # ep_per_fit = 250
    # for ep_per_fit in [50, 150, 250]:
    #     n_epochs = n_samples // ep_per_fit
    #     eps = 0.7
    #     launcher.add_experiment(alg='REPS', distribution='cholesky', eps=eps, n_epochs=n_epochs, ep_per_fit=ep_per_fit)


    # for ep_per_fit in [50, 150, 250]:
    #     n_epochs = n_samples // ep_per_fit
    #     for eps in [0.7, 0.9, 1.1]:
    #         launcher.add_experiment(alg='REPS', distribution='diag', eps=eps, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    # for ep_per_fit in [50, 150, 250]:
    #     n_epochs = n_samples // ep_per_fit
    #     for eps in [0.7, 0.9, 1.1]:
    #         for k in [25, 50, 75]:
    #             for method in ['Pearson', 'MI']:
    #                 for gama in [0.1, 0.5, 0.9]:
    #                     launcher.add_experiment(alg='REPS_MI_full', distribution='mi', eps=eps, k=k, gamma=gama, sample_type='percentage', method=method, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #             launcher.add_experiment(alg='REPS_MI_full', distribution='mi', eps=eps, k=k, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    # n_samples = 5000

    # eps = 4.7
    # kappa = 17.
    # ep_per_fit = 150
    # n_epochs = n_samples // ep_per_fit
    # launcher.add_experiment(alg='ConstrainedREPS', distribution='diag', eps=eps, kappa=kappa, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    # launcher.add_experiment(alg='ConstrainedREPS', distribution='cholesky', eps=eps, kappa=kappa, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    
    # ep_per_fit = 50
    # n_epochs = n_samples // ep_per_fit
    # for method in ['MI', 'Pearson']:
    #     launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=eps, kappa=kappa, k=50, method=method, sample_type='percentage', gamma=0.1, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #     launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=eps, kappa=kappa, k=50, method=method, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    # for ep_per_fit in [25, 50, 150, 250]:
    #     n_epochs = n_samples // ep_per_fit
    #     launcher.add_experiment(alg='ConstrainedREPS', distribution='cholesky', eps=eps, kappa=kappa, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #     for k in [25, 50, 75]:
    #             for method in ['Pearson', 'MI']:
    #                 for gama in [0.1, 0.5, 0.9]:
    #                     launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=eps, kappa=kappa, k=k, gamma=gama, sample_type='percentage', method=method, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #             launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=eps, kappa=kappa, k=k, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)



    # # lqr_ablation_paper
    # n_samples = 5000
    
    # for method in ['Pearson', 'MI']:
    #     for gama in [0.1, 0.3, 0.5, 0.7, 0.9]:
    #         ep_per_fit = 25
    #         n_epochs = n_samples // ep_per_fit
    #         launcher.add_experiment(alg='REPS_MI', distribution='diag', eps=0.4, k=30, method=method, sample_type='percentage', gamma=gama, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
            # launcher.add_experiment(alg='ConstrainedREPSMI', distribution='diag', eps=2.5, kappa=6., k=30, method=method, sample_type='percentage', gamma=gama, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

            # # ep_per_fit = 50
            # ep_per_fit = 250
            # n_epochs = n_samples // ep_per_fit
            # for k in [10, 30, 50]:
            #     launcher.add_experiment(alg='REPS_MI_full', distribution='mi', eps=0.5, k=k, method=method, sample_type='percentage', gamma=gama, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
            #     # launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=4.7, kappa=17., k=k, method=method, sample_type='percentage', gamma=gama, n_epochs=n_epochs, ep_per_fit=ep_per_fit)


    # lqr_ablation_paper_random
    # n_samples = 5000
    # ep_per_fit = 50
    # n_epochs = n_samples // ep_per_fit

    # for method in ['MI', 'Pearson', 'Random']:
    #     launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=4.7, kappa=17., k=50, method=method, sample_type='percentage', gamma=0.1, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #     launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=4.7, kappa=17., k=50, method=method, sample_type=None, gamma=0.1, n_epochs=n_epochs, ep_per_fit=ep_per_fit)


    # n_samples = 15000
    
    # ep_per_fit = 50
    # n_epochs = n_samples // ep_per_fit
    # # launcher.add_experiment(alg='REPS', distribution='diag', eps=0.4, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    # # launcher.add_experiment(alg='ConstrainedREPS', distribution='diag', eps=2.5, kappa=2.0, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    # launcher.add_experiment(alg='ConstrainedREPSMI', distribution='diag', eps=1.5, kappa=2.0, sample_type='percentage', gamma=0.1, k=30, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    
        # launcher.add_experiment(alg='REPS', distribution='cholesky', eps=0.4, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    # for eps in [3.5, 4.5]:
    #     for kappa in [20.]:
    #         ep_per_fit = 200
    #         n_epochs = n_samples // ep_per_fit
    #         launcher.add_experiment(alg='ConstrainedREPS', distribution='cholesky', eps=eps, kappa=kappa, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #         # launcher.add_experiment(alg='ConstrainedREPS', distribution='mi', eps=eps, kappa=kappa, sample_type='percentage', method='MI', k=30, gamma=0.1, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #         ep_per_fit = 50
    #         n_epochs = n_samples // ep_per_fit
    #         launcher.add_experiment(alg='ConstrainedREPS', distribution='diag', eps=eps, kappa=kappa, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #         # launcher.add_experiment(alg='ConstrainedREPS', distribution='mi', eps=eps, kappa=kappa, sample_type='percentage', method='MI', k=30, gamma=0.1, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    
    #         for ep_per_fit in [150, 200]:
    #             n_epochs = n_samples // ep_per_fit
    #             launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=eps, kappa=kappa, sample_type='percentage', gamma=0.1, k=30, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #             launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=eps, kappa=kappa, sample_type='percentage', gamma=0.5, k=30, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
    #             launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=eps, kappa=kappa, sample_type='percentage', gamma=0.9, k=30, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    launcher.add_default_params(lqr_dim=lqr_dim, n_ineff=lqr_dim-eff, env_seed=env_seed, 
                                fit_per_epoch=1,
                                sigma_init=3e-1)

    n_samples = 5000
    
    # # for ep_per_fit in [50, 100, 150, 200, 250]:
    for ep_per_fit in [100]:
        n_epochs = n_samples // ep_per_fit
        for eps in [3e-2, 3e-3, 3e-4]:        
            launcher.add_experiment(alg='PPO', eps=eps, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
            for kappa in [1e-0, 1e-1, 1e-2]:
                launcher.add_experiment(alg='TRPO', eps=eps, kappa=kappa, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
        for eps in [1e-1, 1e-2, 1e-3]:
            launcher.add_experiment(alg='REINFORCE', eps=eps, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    ep_per_fit = 50
    n_epochs = n_samples // ep_per_fit
    launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=4.7, kappa=17., sample_type='percentage', method='Pearson', gamma=0.1, k=50, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    print(experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)

    