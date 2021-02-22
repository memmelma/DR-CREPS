from itertools import product
from experiment_launcher import Launcher
from mushroom_rl.environments import LQR
from mushroom_rl.algorithms.policy_search.black_box_optimization.reps import REPS

if __name__ == '__main__':

    local = False
    test = True

    launcher = Launcher(exp_name='lqr_launcher_7d_500e',
                        python_file='lqr_mi_el',
                        n_exp=25,
                        memory=8000,
                        days=0,
                        hours=5,
                        minutes=0,
                        seconds=0,
                        n_jobs=10,
                        use_timestamp=True)
    
    launcher.add_default_params(eps=0.2, k=7, lqr_dim=7, n_epochs=500, fit_per_epoch=10, ep_per_fit=100)

    launcher.add_experiment(alg='REPS')
    launcher.add_experiment(alg='REPS_MI')

    launcher.run(local, test)
