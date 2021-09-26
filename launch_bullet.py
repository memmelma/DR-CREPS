from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

    local = False
    test = False

    env_seed = 0

    experiment_name = f'bullet_halfcheetah'

    launcher = Launcher(experiment_name,
                        'el_bullet_mi',
                        1,
                        memory=2500,
                        days=3,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=False)
    
    launcher.add_default_params(env_name='HalfCheetahBulletEnv-v0', horizon=0, env_gamma=0.99, # Walker2DBulletEnv-v0, HopperBulletEnv-v0
                                # n_epochs=25, ep_per_fit=3000, fit_per_epoch=4,
                                fit_per_epoch=1, # n_epochs=70
                                sigma_init=3e-1, # 3e-1 maybe even less
                                bins=4, mi_type='regression')
    
    # launcher.add_experiment(alg='REPS', eps=0.2, distribution='cholesky')
    # launcher.add_experiment(alg='RWR', eps=0.2, distribution='cholesky')
    # launcher.add_experiment(alg='ConstrainedREPS', eps=3., kappa=2., distribution='cholesky')

    # for k in [50, 100, 200]:
    # for k in [100, 300, 500]:
        # for eps in [0.5, 0.7, 0.9]:
        # for eps in [0.3, 0.6, 0.9]:
        #         launcher.add_experiment(alg='REPS_MI_full', eps=eps, method='MI', sample_type='percentage', gamma=0.3, k=k, distribution='mi')
        #         launcher.add_experiment(alg='REPS_MI_full', eps=eps, method='MI', sample_type='percentage', gamma=0.6, k=k, distribution='mi')
        #         launcher.add_experiment(alg='REPS_MI_full', eps=eps, method='MI', sample_type='percentage', gamma=0.9, k=k, distribution='mi')

    for ep_per_fit in [550, 650, 750]:
        n_epochs = int((70*750) // ep_per_fit)
        for k in [250, 350]:
            for eps in [0.2, 0.3, 0.4]:
                launcher.add_experiment(alg='REPS_MI_full', eps=eps, method='MI', sample_type='percentage', gamma=0.5, k=k, distribution='mi')
                launcher.add_experiment(alg='REPS_MI_full', eps=eps, method='MI', sample_type='percentage', gamma=0.6, k=k, distribution='mi')
                launcher.add_experiment(alg='REPS_MI_full', eps=eps, method='MI', sample_type='percentage', gamma=0.7, k=k, distribution='mi')

    # for k in [50, 100, 250]:
    #     for eps in [5., 10.]:
    #         for kappa in [5., 20.]:
    #             launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=eps, kappa=kappa, method='MI', sample_type='percentage', gamma=0.3, k=k, distribution='mi')
    #             launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=eps, kappa=kappa, method='MI', sample_type='percentage', gamma=0.6, k=k, distribution='mi')
    #             launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=eps, kappa=kappa, method='MI', sample_type='percentage', gamma=0.9, k=k, distribution='mi')

    #     launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=3., kappa=2., method='MI', sample_type='percentage', gamma=0.1, k=k, distribution='mi')

    #     launcher.add_experiment(alg='REPS_MI', eps=0.2, method='MI', sample_type='percentage', gamma=0.9, k=k, distribution='diag')
    #     launcher.add_experiment(alg='REPS_MI', eps=0.2, method='Pearson', sample_type='percentage', gamma=0.9, k=k, distribution='diag')

    #     launcher.add_experiment(alg='REPS_MI_full', eps=0.2, method='MI', sample_type='percentage', gamma=0.9, k=k, distribution='mi')
    #     launcher.add_experiment(alg='REPS_MI_full', eps=0.2, method='Pearson', sample_type='percentage', gamma=0.9, k=k, distribution='mi')

    #     launcher.add_experiment(alg='ConstrainedREPSMI', eps=3., kappa=2., method='MI', sample_type='percentage', gamma=0.1, k=k, distribution='diag')
    #     launcher.add_experiment(alg='ConstrainedREPSMI', eps=3., kappa=2., method='Pearson', sample_type='percentage', gamma=0.1, k=k, distribution='diag')

    #     launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=3., kappa=2., method='MI', sample_type='percentage', gamma=0.1, k=k, distribution='mi')
    #     launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=3, kappa=2., method='Pearson', sample_type='percentage', gamma=0.1, k=k, distribution='mi')

    #     launcher.add_experiment(alg='ConstrainedREPSMI', eps=1., kappa=2., method='MI', sample_type='percentage', gamma=0.1, k=k, distribution='diag')
    #     launcher.add_experiment(alg='ConstrainedREPSMI', eps=1., kappa=2., method='Pearson', sample_type='percentage', gamma=0.1, k=k, distribution='diag')

    #     launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=1., kappa=2., method='MI', sample_type='percentage', gamma=0.1, k=k, distribution='mi')
    #     launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=1, kappa=2., method='Pearson', sample_type='percentage', gamma=0.1, k=k, distribution='mi')

    ### TWO DIFFERENT sigmas 3e-1 3e-1
    # # Constrained REPS all bad / learn way to slow
    # launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=1., kappa=1., method='MI', sample_type='percentage', gamma=0.1, k=100, distribution='mi')
    # launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=.5, kappa=1., method='MI', sample_type='percentage', gamma=0.1, k=400, distribution='mi')
    # launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=.5, kappa=1., method='Pearson', sample_type='percentage', gamma=0.1, k=400, distribution='mi')
    # # REPS and RWR die after couple of iterations -> updates too large
    # launcher.add_experiment(alg='REPS', eps=0.02, distribution='cholesky')
    # launcher.add_experiment(alg='RWR', eps=0.02, distribution='cholesky')
    # launcher.add_experiment(alg='REPS_MI_full', eps=0.5, method='MI', sample_type='percentage', gamma=0.9, k=400, distribution='mi')
    # launcher.add_experiment(alg='REPS_MI_full', eps=1., method='MI', sample_type='percentage', gamma=0.9, k=400, distribution='mi')
    # launcher.add_experiment(alg='REPS_MI', eps=0.5, method='MI', sample_type='percentage', gamma=0.9, k=100, distribution='diag')
    # launcher.add_experiment(alg='REPS_MI', eps=0.5, method='Pearson', sample_type='percentage', gamma=0.9, k=100, distribution='diag') # eps too low
    # launcher.add_experiment(alg='REPS_MI', eps=1., method='MI', sample_type='percentage', gamma=0.9, k=100, distribution='diag') # eps too high

    # launcher.add_experiment(alg='REPS_MI', eps=.7, method='MI', sample_type='percentage', gamma=0.9, k=100, distribution='diag')
    # launcher.add_experiment(alg='REPS_MI', eps=.7, method='Pearson', sample_type='percentage', gamma=0.9, k=100, distribution='diag')

    # launcher.add_experiment(alg='REPS_MI', eps=.9, method='MI', sample_type='percentage', gamma=0.9, k=100, distribution='diag')
    # launcher.add_experiment(alg='REPS_MI', eps=.9, method='Pearson', sample_type='percentage', gamma=0.9, k=100, distribution='diag')

    # launcher.add_experiment(alg='REPS_MI', eps=.9, method='MI', sample_type='percentage', gamma=0.9, k=75, distribution='diag')
    # launcher.add_experiment(alg='REPS_MI', eps=.9, method='Pearson', sample_type='percentage', gamma=0.9, k=75, distribution='diag')

    # launcher.add_experiment(alg='REPS_MI', eps=.9, method='MI', sample_type='percentage', gamma=0.9, k=125, distribution='diag')
    # launcher.add_experiment(alg='REPS_MI', eps=.9, method='Pearson', sample_type='percentage', gamma=0.9, k=125, distribution='diag')

    # launcher.add_experiment(alg='REPS_MI', eps=1.1, method='MI', sample_type='percentage', gamma=0.9, k=100, distribution='diag')
    # launcher.add_experiment(alg='REPS_MI', eps=1.1, method='Pearson', sample_type='percentage', gamma=0.9, k=100, distribution='diag')
    
    # launcher.add_experiment(alg='REPS_MI', eps=.7, method='MI', sample_type='percentage', gamma=0.9, k=50, distribution='diag')
    # launcher.add_experiment(alg='REPS_MI', eps=.7, method='Pearson', sample_type='percentage', gamma=0.9, k=50, distribution='diag')

    # launcher.add_experiment(alg='REPS_MI_full', eps=0.5, method='MI', sample_type='percentage', gamma=0.9, k=100, distribution='mi')
    # launcher.add_experiment(alg='REPS_MI_full', eps=1., method='MI', sample_type='percentage', gamma=0.9, k=100, distribution='mi')
    
    # launcher.add_experiment(alg='REPS_MI_full', eps=0.5, method='Pearson', sample_type='percentage', gamma=0.9, k=100, distribution='mi')
    # launcher.add_experiment(alg='REPS_MI_full', eps=1., method='Pearson', sample_type='percentage', gamma=0.9, k=100, distribution='mi')

    # launcher.add_experiment(alg='REPS', eps=0.005, distribution='diag')
    # launcher.add_experiment(alg='REPS', eps=0.005, distribution='cholesky')

    print(experiment_name)
    print('experiments:', len(launcher._experiment_list))

    launcher.run(local, test)