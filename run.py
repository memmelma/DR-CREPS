from experiment_launcher import Launcher
from experiment_config import default_params

if __name__ == '__main__':

    local = True
    test = False

    experiment_name = f'test'

    launcher = Launcher(experiment_name,
                        'experiment_config',
                        5,
                        memory=500,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=False,
                        conda_env='iprl'
                        )
    
    launcher.add_default_params(**default_params())

    launcher.add_experiment(
        env='LQR',
        alg='DR-CREPS-PE', eps=4.7, kappa=17., k=50,
        distribution='gdr',
        C='PCC',
        sample_strat='percentage', lambd=0.1,
        n_epochs=10, ep_per_fit=50
    )

    print('name:', experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)