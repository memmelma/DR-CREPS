from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

    local = False
    test = False

    env_seed = 0
    lqr_dim = 10
    eff = 3

    experiment_name = f'lqr_dim_{lqr_dim}{"_eff_" + str(eff) + "_env_" + str(env_seed) if env_seed >= 0 else ""}_hp_sample'

    launcher = Launcher(exp_name=experiment_name,
                        python_file='el_lqr_mi',
                        n_exp=25,
                        memory=500, # for 10 dim
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        n_jobs=-1,
                        use_timestamp=True)
    
    launcher.add_default_params(lqr_dim=lqr_dim, n_epochs=100, fit_per_epoch=1, ep_per_fit=25, env_seed=env_seed, n_ineff=lqr_dim-eff,
                                sigma_init=3e-1, distribution = 'diag', # 3e-1 for diag, use np.sqrt(3e-1)=1e-1 for others
                                bins=4, mi_type='regression')
    
    # launcher.add_experiment(alg='RWR', eps=0.7, sample_type='None')
    # launcher.add_experiment(alg='REPS', eps=0.7, sample_type='None')
    # launcher.add_experiment(alg='REPS_MI', eps=0.7, sample_type='percentage', gamma=1e-1, k=30)
    # launcher.add_experiment(alg='ConstrainedREPS', eps=0.7, kappa=4.0, sample_type='None')
    # launcher.add_experiment(alg='ConstrainedREPSMI', eps=0.7, kappa=4.0, sample_type='percentage', gamma=1e-1, k=30)
    # launcher.add_experiment(alg='REPS_MI', eps=0.7, sample_type='importance')
    # launcher.add_experiment(alg='ConstrainedREPSMI', eps=0.7, kappa=4.0, sample_type='importance', k=30)

    # launcher.add_experiment(alg='REPS_MI', eps=0.7, sample_type='importance')
    # launcher.add_experiment(alg='ConstrainedREPSMI', eps=0.7, kappa=4.0, gamma=89., sample_type='importance', k=30)


    # MORE debug
    # eps = 1.0
    # launcher.add_experiment(alg='REPS', eps=eps)
    # launcher.add_experiment(alg='RWR', eps=eps)
    # launcher.add_experiment(alg='ConstrainedREPS', eps=eps)

    # for eps in range(1, 8, 1):
    #     for kappa in range(50, 1500, 100):
    #         launcher.add_experiment(alg='MORE', eps=round(eps,1), kappa=round(kappa,1))

    # launcher.add_experiment(alg='MORE', kappa=1000, eps=0.5, sample_type='90p')
    # launcher.add_experiment(alg='MORE', kappa=1000, eps=1.0, sample_type='90p')
    # launcher.add_experiment(alg='MORE', kappa=1000, eps=1.5, sample_type='90p')
    # launcher.add_experiment(alg='MORE', kappa=1000, eps=2.0, sample_type='90p')
    # launcher.add_experiment(alg='MORE', kappa=1000, eps=2.5, sample_type='90p')
    # launcher.add_experiment(alg='MORE', kappa=1000, eps=3.0, sample_type='90p')
    # launcher.add_experiment(alg='MORE', kappa=1000, eps=4.0, sample_type='90p')
    # launcher.add_experiment(alg='MORE', kappa=1000, eps=5.0, sample_type='90p')

    # launcher.add_experiment(alg='MORE', eps=0.5)
    # launcher.add_experiment(alg='MORE', eps=1.0)
    # launcher.add_experiment(alg='MORE', eps=2.0)
    # launcher.add_experiment(alg='MORE', eps=3.0)
    # launcher.add_experiment(alg='MORE', eps=4.0)
    # launcher.add_experiment(alg='MORE', eps=5.0)
    # launcher.add_experiment(alg='MORE', eps=6.0)
    # launcher.add_experiment(alg='MORE', eps=7.0)
    # launcher.add_experiment(alg='MORE', eps=8.0)
    # launcher.add_experiment(alg='MORE', eps=9.0)
    # launcher.add_experiment(alg='MORE', eps=10.0)


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
    # hyperparameters according to max reward
    for k in range(5, 100, 5):
        eps = 2.7
        kappa = 5.0
        launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type=None, eps=eps, kappa=kappa)
        launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='importance', eps=eps, kappa=kappa)
        
        for gama in np.arange(0.1, 1.0, 0.1):
            gama = round(gama,1)
            launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='percentage', gamma=gama, eps=eps, kappa=kappa)

        # eps = 0.5
        # launcher.add_experiment(alg='REPS_MI', k=k, sample_type=None, eps=eps)
        # launcher.add_experiment(alg='REPS_MI', k=k, sample_type='importance', eps=eps)

        # for gama in np.arange(0.1, 1.0, 0.1):
        #     gama = round(gama,1)
        #     launcher.add_experiment(alg='REPS_MI',k=k, sample_type='percentage', gamma=gama, eps=eps)

    ### ### ###
        
    # sigma and sample_type -> MISSING k !!!111elfelven
    # for k in range(5, 100, 10):
    #     eps = 2.6
    #     kappa = 4.0
    #     launcher.add_experiment(alg='ConstrainedREPSMI', eps=eps, kappa=kappa, k=k, sample_type=None)
    #     launcher.add_experiment(alg='ConstrainedREPSMI', eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=-1) # linear parameter 1->0
    #     launcher.add_experiment(alg='ConstrainedREPSMI', eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=-2) # linear parameter 0->1
    #     for gama in np.arange(0.1, 1.0, 0.1):
    #         launcher.add_experiment(alg='ConstrainedREPSMI', eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=gama)
        
        # eps = 0.6
        # launcher.add_experiment(alg='REPS_MI', eps=eps, k=k, sample_type=None)
        # launcher.add_experiment(alg='REPS_MI', eps=eps, k=k, sample_type='percentage', gamma=-1) # linear parameter 1->0
        # launcher.add_experiment(alg='REPS_MI', eps=eps, k=k, sample_type='percentage', gamma=-2) # linear parameter 0->1
        # for gama in np.arange(0.1, 1.0, 0.1):
        #     gama = round(gama,1)
        #     launcher.add_experiment(alg='REPS_MI', eps=eps, k=k, sample_type='percentage', gamma=gama)


    # launcher.add_experiment(alg='REPS', eps=0.1)
    # launcher.add_experiment(alg='REPS', eps=0.3)
    # launcher.add_experiment(alg='REPS', eps=0.5)
    # launcher.add_experiment(alg='REPS', eps=0.7)
    # launcher.add_experiment(alg='REPS', eps=0.9)
    # launcher.add_experiment(alg='REPS', eps=1.1)

    # launcher.add_experiment(alg='ConstrainedREPS', eps=0.1, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPS', eps=0.3, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPS', eps=0.5, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPS', eps=0.7, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPS', eps=0.9, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPS', eps=1.1, kappa=2)

    # ours vs all
    # launcher.add_experiment(alg='MORE')
    # launcher.add_experiment(alg='RWR')
    # launcher.add_experiment(alg='REPS')
    # launcher.add_experiment(alg='REPS_MI', k=25, gamma=0.9)
    # launcher.add_experiment(alg='ConstrainedREPS')
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=25, gamma=0.1) 
    # launcher.add_experiment(alg='ConstrainedREPSMIOracle', k=25, gamma=0.1)

    # ours vs ours
    # k
    # launcher.add_experiment(alg='ConstrainedREPS', sample_type=None, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=5, sample_type=None, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type=None, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=50, sample_type=None, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=100, sample_type=None, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=250, sample_type=None, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=500, sample_type=None, kappa=2)
    
    # kappa 
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=, sample_type=None, kappa=1)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=, sample_type=None, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=, sample_type=None, kappa=3)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=, sample_type=None, kappa=4)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=, sample_type=None, kappa=5)

    # sample_type, gamma
    # launcher.add_experiment(alg='REPS_MI', k=62, sample_type=None)
    # launcher.add_experiment(alg='REPS_MI', k=62, sample_type='percentage', gamma=0.1)
    # launcher.add_experiment(alg='REPS_MI', k=62, sample_type='percentage', gamma=0.5)
    # launcher.add_experiment(alg='REPS_MI', k=62, sample_type='percentage', gamma=0.9)
    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type=None)
    # launcher.add_experiment(alg='ConstrainedREPSMI', gamma=0.1, k=25, kappa=5)
    


    # kappa
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type=None, kappa=0.5)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type=None, kappa=1)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type=None, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type=None, kappa=3)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type=None, kappa=4)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type=None, kappa=5)

    # launcher.add_experiment(alg='REPS_MI', k=25) 
    # launcher.add_experiment(alg='REPS_MI', k=100)

    # launcher.add_experiment(alg='ConstrainedREPS', kappa=5)

    # launcher.add_experiment(alg='ConstrainedREPSMI', gamma=0.1, k=25, kappa=5)
    # launcher.add_experiment(alg='ConstrainedREPSMI', gamma=0.1, k=100, kappa=5)

    # launcher.add_experiment(alg='ConstrainedREPS', kappa=2)

    # launcher.add_experiment(alg='REPS_MI', k=5, mi_avg=1)
    # launcher.add_experiment(alg='REPS_MI', k=5, mi_avg=0)

    # launcher.add_experiment(alg='REPS')
    # launcher.add_experiment(alg='REPS_MI', k=25, gamma=0.99) 
    # launcher.add_experiment(alg='REPS_MI', k=25, gamma=-1)
    # launcher.add_experiment(alg='REPS_MI', k=25, gamma=0.1) 
    # launcher.add_experiment(alg='REPS_MI', k=25, gamma=0.9) 

    # launcher.add_experiment(alg='ConstrainedREPS')
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=25, gamma=-1)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=25, gamma=0.1) 
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=25, gamma=0.9) 

    # launcher.add_experiment(alg='ConstrainedREPSMI', gamma=0.1, kappa=2, k=5, mi_avg=1)
    # launcher.add_experiment(alg='ConstrainedREPSMI', gamma=0.1, kappa=2, k=5, mi_avg=0)

    # launcher.add_experiment(alg='ConstrainedREPSMI', gamma=0.1, kappa=2, k=25, mi_avg=1)
    # launcher.add_experiment(alg='ConstrainedREPSMI', gamma=0.1, kappa=2, k=25, mi_avg=0)

    # launcher.add_experiment(alg='ConstrainedREPSMI', gamma=0.1, k=100, kappa=2)

    # launcher.add_experiment(alg='REPS', sample_type=None)
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=0.1)
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=0.2)
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=0.3)
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=0.5)
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=0.6)
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=0.7)
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=0.8)
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=0.9)
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1.0)

    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='percentage', gamma=0.1)
    # launcher.add_experiment(alg='ConstrainedREPS')

    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='fixed', gamma=1e-10)
    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='fixed', gamma=1e-5)
    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='fixed', gamma=1e-3)

    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='percentage', gamma=1e-1)
    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='percentage', gamma=2e-1)
    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='percentage', gamma=5e-1)

    # launcher.add_experiment(alg='ConstrainedREPSMI')
    # launcher.add_experiment(alg='ConstrainedREPSMIOracle', sample_type='fixed', gamma=1e-5)
    # launcher.add_experiment(alg='ConstrainedREPSMIOracle', sample_type='percentage', gamma=1e-1)
    # launcher.add_experiment(alg='ConstrainedREPS')

    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=5)
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=11) # lin 1- 0 GREAT
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=12) # lin 0.1 
    
    # launcher.add_experiment(alg='REPS_MI_ORACLE', sample_type='percentage', gamma=1e-1, kappa=12)
    # launcher.add_experiment(alg='REPS_MI_ORACLE', sample_type='percentage', gamma=9e-1, kappa=12)
    # launcher.add_experiment(alg='REPS_MI_ORACLE', sample_type='percentage', gamma=1e-1, kappa=13) # lin

    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=9e-1, kappa=12) # lin 0.1 
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1., kappa=12) # lin 0.1 
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=.99, kappa=12) # lin 0.1 
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=12) # lin 0 - 1 
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=66) # exp 1 - 0 GREAT
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=56) # exp 0 - 1
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=13) # lin 0.5 - 0
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=57) # exp 0.5 - 0
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=88) # lin + exp /2
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=98) # exp 0.9 - 0.0 + 0.1
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=58, avg_mi=True) # exp 
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=59, avg_mi=False) # exp 
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=14, avg_mi=True) # lin
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=15, avg_mi=False) # lin
    # launcher.add_experiment(alg='REPS_MI', sample_type='percentage', gamma=1e-1, kappa=16) # lin 0.9 - 0.0

    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='percentage', gamma=1e-1, kappa=2)
    # launcher.add_experiment(alg='ConstrainedREPSMIOracle')

    # launcher.add_experiment(alg='MORE')
    # launcher.add_experiment(alg='REPS')
    # launcher.add_experiment(alg='REPS_MI')
    # launcher.add_experiment(alg='RWR')
    # launcher.add_experiment(alg='RWR', eps=0.7)
    # launcher.add_experiment(alg='ConstrainedREPS')
    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='percentage', gamma=1e-1)
    
    # launcher.add_experiment(alg='ConstrainedREPSMIOracle', k=k)
    # launcher.add_experiment(alg='ConstrainedREPSMIOracle', k=k, sample_type='fixed', gamma=1e-5)
    # launcher.add_experiment(alg='ConstrainedREPSMIOracle', k=2*k)
    # launcher.add_experiment(alg='ConstrainedREPSMIOracle', k=2*k, sample_type='fixed', gamma=1e-5)

    # launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='fixed', gamma=1e-5)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='percentage', gamma=0.1)

    # launcher.add_experiment(alg='ConstrainedREPSMI', k=2*k, sample_type='fixed', gamma=1e-5)
    # launcher.add_experiment(alg='ConstrainedREPSMI', k=2*k, sample_type='percentage', gamma=0.1)


    print(experiment_name)

    launcher.run(local, test)