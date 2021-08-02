from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

    local = False
    test = False

    env_seed = 0
    lqr_dim = 10
    eff = 3

    experiment_name = f'lqr_dim_{lqr_dim}{"_eff_" + str(eff) + "_env_" + str(env_seed) if env_seed >= 0 else ""}_more'

    launcher = Launcher(experiment_name,
                        'el_lqr_mi',
                        25,
                        memory=500, # for 10 dim
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True)
    
    launcher.add_default_params(lqr_dim=lqr_dim, fit_per_epoch=1, env_seed=env_seed, n_ineff=lqr_dim-eff,# n_epochs=100, ep_per_fit=25,
                                sigma_init=3e-1, #distribution = 'diag', # 3e-1 for diag
                                bins=4, mi_type='regression')
    

    # launcher.add_experiment(alg='ConstrainedREPS', distribution='diag', ep_per_fit=250, n_epochs=200, eps=2.7)
    # launcher.add_experiment(alg='ConstrainedREPS', distribution='cholesky', ep_per_fit=250, n_epochs=200, eps=2.7)

    # launcher.add_experiment(alg='ConstrainedREPSMIFull', k=10, distribution='mi', ep_per_fit=250, n_epochs=200, eps=2.7)
    # launcher.add_experiment(alg='ConstrainedREPSMIFull', k=30, distribution='mi', ep_per_fit=250, n_epochs=200, eps=2.7)
    # launcher.add_experiment(alg='ConstrainedREPSMIFull', k=50, distribution='mi', ep_per_fit=250, n_epochs=200, eps=2.7)

    # launcher.add_experiment(alg='ConstrainedREPSMIFull', k=10, distribution='mi', ep_per_fit=25, n_epochs=200, eps=2.7)
    # launcher.add_experiment(alg='ConstrainedREPSMIFull', k=30, distribution='mi', ep_per_fit=25, n_epochs=200, eps=2.7)
    # launcher.add_experiment(alg='ConstrainedREPSMIFull', k=50, distribution='mi', ep_per_fit=25, n_epochs=200, eps=2.7)
    
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=10, distribution='diag', ep_per_fit=250, n_epochs=200, eps=2.7)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=30, distribution='diag', ep_per_fit=250, n_epochs=200, eps=2.7)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=50, distribution='diag', ep_per_fit=250, n_epochs=200, eps=2.7)
    
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=10, distribution='diag', ep_per_fit=25, n_epochs=200, eps=2.7)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=30, distribution='diag', ep_per_fit=25, n_epochs=200, eps=2.7)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=50, distribution='diag', ep_per_fit=25, n_epochs=200, eps=2.7)


    # # full covariance
    # launcher.add_experiment(alg='REPS', distribution='diag', ep_per_fit=250, n_epochs=200, eps=.5)
    launcher.add_experiment(alg='RWR', distribution='diag', ep_per_fit=250, n_epochs=200, eps=.5)
    # launcher.add_experiment(alg='ConstrainedREPS', distribution='diag', ep_per_fit=250, n_epochs=200, eps=2.7)

    # launcher.add_experiment(alg='REPS', distribution='cholesky', ep_per_fit=250, n_epochs=200, eps=.5)
    # launcher.add_experiment(alg='RWR', distribution='cholesky', ep_per_fit=250, n_epochs=200, eps=.5)
    # launcher.add_experiment(alg='ConstrainedREPS', distribution='cholesky', ep_per_fit=250, n_epochs=200, eps=1.5, kappa=5.)

    launcher.add_experiment(alg='MORE', distribution='cholesky', ep_per_fit=250, n_epochs=200, eps=5., kappa=.5)
    launcher.add_experiment(alg='MORE', distribution='cholesky', ep_per_fit=250, n_epochs=200, eps=5., kappa=.2)
    launcher.add_experiment(alg='MORE', distribution='cholesky', ep_per_fit=250, n_epochs=200, eps=5., kappa=.1)

    launcher.add_experiment(alg='MORE', distribution='cholesky', ep_per_fit=250, n_epochs=200, eps=1., kappa=1.)
    launcher.add_experiment(alg='MORE', distribution='cholesky', ep_per_fit=250, n_epochs=200, eps=2., kappa=1.)
    launcher.add_experiment(alg='MORE', distribution='cholesky', ep_per_fit=250, n_epochs=200, eps=3., kappa=1.)

    # launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', ep_per_fit=25, n_epochs=200, eps=2.7, kappa=5., k=85)
    # launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', ep_per_fit=250, n_epochs=200, eps=2.7, kappa=5., k=85)
    # # launcher.add_experiment(alg='ConstrainedREPS', distribution='cholesky', ep_per_fit=250, n_epochs=200, eps=5., kappa=5.) -> crash

    # launcher.add_experiment(alg='ConstrainedREPSMIFull', sample_type='percentage', gamma=0.1, distribution='mi', ep_per_fit=250, n_epochs=200, eps=2.7, kappa=5., k=85)
    # launcher.add_experiment(alg='REPS_MI_full', sample_type='percentage', gamma=0.9, distribution='mi', ep_per_fit=250, n_epochs=200, eps=.5, k=45)


    # # eps and kappa:
    # for eps in np.arange(0.3, 5.0, 0.2):
    #     eps = round(eps,1)
    #     # REPS
    #     launcher.add_experiment(alg='REPS', eps=eps)
    #     # RWR
    #     launcher.add_experiment(alg='RWR', eps=eps)

    #     for kappa in np.arange(1., 10., 1.):
    #         kappa = round(kappa,1)
    #         # Constrained REPS
    #         launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa)

    # k -> lqr w/ 10dim has 100 parameters
    # # hyperparameters according to max reward
    # for k in range(5, 100, 5):
    #     eps = 2.7
    #     kappa = 5.0
    #     launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type=None, eps=eps, kappa=kappa)
    #     launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='importance', eps=eps, kappa=kappa)
        
    #     for gama in np.arange(0.1, 1.0, 0.1):
    #         gama = round(gama,1)
    #         launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='percentage', gamma=gama, eps=eps, kappa=kappa)

    #     eps = 0.5
    #     launcher.add_experiment(alg='REPS_MI', k=k, sample_type=None, eps=eps)
    #     launcher.add_experiment(alg='REPS_MI', k=k, sample_type='importance', eps=eps)

    #     for gama in np.arange(0.1, 1.0, 0.1):
    #         gama = round(gama,1)
    #         launcher.add_experiment(alg='REPS_MI',k=k, sample_type='percentage', gamma=gama, eps=eps)

    # hyperparameters according to min regret
    # for k in range(5, 100, 5):
        # eps = 3.1
        # kappa = 6.0
        # launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type=None, eps=eps, kappa=kappa)
        # launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='importance', eps=eps, kappa=kappa)
        
        # for gama in np.arange(0.1, 1.0, 0.1):
        #     gama = round(gama,1)
        #     launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='percentage', gamma=gama, eps=eps, kappa=kappa)

        # eps = 3.1
        # launcher.add_experiment(alg='REPS_MI', k=k, sample_type=None, eps=eps)
        # launcher.add_experiment(alg='REPS_MI', k=k, sample_type='importance', eps=eps)

        # for gama in np.arange(0.1, 1.0, 0.1):
        #     gama = round(gama,1)
        #     launcher.add_experiment(alg='REPS_MI',k=k, sample_type='percentage', gamma=gama, eps=eps)

    # # MI vs Pearson
    # for sample_type in ['PRO', 'importance', 'percentage']:
    #     for method in ['MI', 'Pearson']:
    #         if sample_type == 'percentage':
    #                 launcher.add_experiment(alg='ConstrainedREPSMI', sample_type=sample_type, method=method, gamma=0.1, k=85, kappa=5.0, eps=2.7)
    #                 launcher.add_experiment(alg='REPS_MI', sample_type=sample_type, method=method, gamma=0.9, k=45, eps=0.5) 
    #         else:
    #             launcher.add_experiment(alg='ConstrainedREPSMI', sample_type=sample_type, method=method, k=85, kappa=5.0, eps=2.7)
    #             launcher.add_experiment(alg='REPS_MI', sample_type=sample_type, method=method, k=45, eps=0.5)

    print(experiment_name)

    launcher.run(local, test)