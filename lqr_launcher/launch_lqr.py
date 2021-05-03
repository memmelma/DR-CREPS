from itertools import product
from experiment_launcher import Launcher

if __name__ == '__main__':

    local = False
    test = False

    env_seed = 0
    lqr_dim = 10
    eff = 3
    k = eff*lqr_dim

    experiment_name = f'lqr_dim_{lqr_dim}{"_eff_" + str(eff) + "_env_" + str(env_seed) if env_seed >= 0 else ""}_con_sample_oracle'

    launcher = Launcher(exp_name=experiment_name,
                        python_file='lqr_mi_el',
                        n_exp=25,
                        memory=5000,
                        days=0,
                        hours=5,
                        minutes=0,
                        seconds=0,
                        n_jobs=25,
                        use_timestamp=True)
    
    launcher.add_default_params(k=k, lqr_dim=lqr_dim, n_epochs=75, fit_per_epoch=1, ep_per_fit=100, env_seed=env_seed, n_ineff=lqr_dim-eff, kappa=2, sigma_init=1e-1, eps=0.7)
    
    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='fixed', gamma=1e-10)
    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='fixed', gamma=1e-5)
    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='fixed', gamma=1e-3)

    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='percentage', gamma=1e-1)
    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='percentage', gamma=2e-1)
    # launcher.add_experiment(alg='ConstrainedREPSMI', sample_type='percentage', gamma=5e-1)

    # launcher.add_experiment(alg='ConstrainedREPSMI')
    launcher.add_experiment(alg='ConstrainedREPSMIOracle', sample_type='fixed', gamma=1e-5)
    launcher.add_experiment(alg='ConstrainedREPSMIOracle', sample_type='percentage', gamma=1e-1)
    # launcher.add_experiment(alg='ConstrainedREPS')

    # launcher.add_experiment(alg='ConstrainedREPS')
    # launcher.add_experiment(alg='ConstrainedREPSMI')
    # launcher.add_experiment(alg='ConstrainedREPSMIOracle')

    print(experiment_name)

    launcher.run(local, test)
