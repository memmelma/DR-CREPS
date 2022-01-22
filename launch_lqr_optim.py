from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

    local = False
    test = False

    env_seed = 0
    lqr_dim = 10
    eff = 3

    experiment_name = f'lqr_nm_bfgs_fix_02'

    launcher = Launcher(experiment_name,
                        'el_lqr_optim',
                        25,
                        memory=500,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=False,
                        conda_env='iprl')
	
    launcher.add_default_params(lqr_dim=lqr_dim, n_ineff=lqr_dim-eff, env_seed=env_seed, 
                                ep_per_fit = 1, fit_per_epoch=1, 
                                sigma_init=3e-1)

    n_samples = 5000
    n_epochs = n_samples
    launcher.add_experiment(alg='NM', n_epochs=n_epochs)
    launcher.add_experiment(alg='BFGS', n_epochs=n_epochs)

    print(experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)

    