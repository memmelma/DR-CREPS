from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

    local = True
    test = False

    env_seed = 0

    experiment_name = f'bulet'

    launcher = Launcher(experiment_name,
                        'el_bullet_mi',
                        2,
                        memory=1000,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True)
    
    launcher.add_default_params(env_name = 'HopperBulletEnv-v0', horizon = None, env_gamma = 0.99, env_seed=env_seed, 
                                n_epochs=50, ep_per_fit=3000, fit_per_epoch=4,
                                sigma_init=3e-1,
                                bins=4, mi_type='regression')
    
    launcher.add_experiment(alg='REPS', eps=0.5, distribution='cholesky')
    launcher.add_experiment(alg='RWR', eps=0.5, distribution='cholesky')
    launcher.add_experiment(alg='ConstrainedREPS', eps=2.5, kappa=2., distribution='cholesky')

    for k in [100, 200, 400]:
        launcher.add_experiment(alg='REPS_MI', eps=0.5, method='MI', sample_type='percentage', gamma=0.9, k=k, distribution='cholesky')
        launcher.add_experiment(alg='REPS_MI', eps=0.5, method='Pearson', sample_type='percentage', gamma=0.9, k=k, distribution='cholesky')

        launcher.add_experiment(alg='ConstrainedREPSMI', eps=2.5, kappa=2., method='MI', sample_type='percentage', gamma=0.1, k=k, distribution='cholesky')
        launcher.add_experiment(alg='ConstrainedREPSMI', eps=2.5, kappa=2., method='Pearson', sample_type='percentage', gamma=0.1, k=k, distribution='cholesky')

        launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=2.5, kappa=2., method='MI', sample_type='percentage', gamma=0.1, k=k, distribution='mi')
        launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=2.5, kappa=2., method='Pearson', sample_type='percentage', gamma=0.1, k=k, distribution='mi')

    print(experiment_name)

    launcher.run(local, test)