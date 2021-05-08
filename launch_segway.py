from itertools import product
from experiment_launcher import Launcher
from mushroom_rl.algorithms.policy_search.black_box_optimization.reps import REPS
from constrained_REPS import REPS_CON
if __name__ == '__main__':

    local = True
    test = False

    k = 2
    
    launcher = Launcher(exp_name=f'segway_{k}',
                        python_file='segway_mi_el',
                        n_exp=1,
                        memory=5000,
                        days=0,
                        hours=5,
                        minutes=0,
                        seconds=0,
                        n_jobs=4,
                        use_timestamp=True)
    
    launcher.add_default_params(eps=0.2, k=k, n_epochs=1, fit_per_epoch=10, ep_per_fit=10)

    launcher.add_experiment(alg='REPS')
    launcher.add_experiment(alg='REPS_MI')
    # launcher.add_experiment(alg='REPS_MI_CON')

    launcher.run(local, test)
