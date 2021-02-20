from itertools import product
from experiment_launcher import Launcher
from mushroom_rl.environments import LQR
from mushroom_rl.algorithms.policy_search.black_box_optimization.reps import REPS

if __name__ == '__main__':

    local = False
    test = False

    launcher = Launcher(exp_name='lqr_launcher',
                        python_file='lqr_mi_el',
                        n_exp=25,
                        memory=2000,
                        days=0,
                        hours=5,
                        minutes=0,
                        seconds=0,
                        n_jobs=5,
                        use_timestamp=True)
    
    launcher.add_default_params(alg='REPS', lqr_dim=3, n_epochs=100, fit_per_epoch=10, ep_per_fit=100, quiet=True)

    eps_ = [0.001, 0.01, 0.1, 0.15, 0.2]

    for eps in eps_:
        launcher.add_experiment(params={'eps': eps})

    launcher.run(local, test)
