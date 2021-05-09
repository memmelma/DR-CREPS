from itertools import product
from experiment_launcher import Launcher

if __name__ == '__main__':

    local = False
    test = False

    experiment_name = f'ball'
    k = 5
    
    launcher = Launcher(exp_name=experiment_name,
                        python_file='ball_mi_el',
                        n_exp=5,
                        memory=5000,
                        days=0,
                        hours=5,
                        minutes=0,
                        seconds=0,
                        n_jobs=25,
                        use_timestamp=True)
    
    launcher.add_default_params(k=k, n_epochs=200, fit_per_epoch=1, ep_per_fit=25, kappa=2, n_basis=20, horizon=1000, sigma_init=1, eps=5)

    launcher.add_experiment(alg='REPS')
    launcher.add_experiment(alg='ConstrainedREPS')
    launcher.add_experiment(alg='ConstrainedREPSMI')
    launcher.add_experiment(alg='ConstrainedREPSMIOracle')
    print(experiment_name)

    launcher.run(local, test)
