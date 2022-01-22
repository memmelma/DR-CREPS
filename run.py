from experiment_launcher import Launcher
from .reproduce_experiments import reproduce_lqr_experiments, reproduce_ship_steering, reproduce_air_hockey, reproduce_ball_stopping


if __name__ == '__main__':

    local = False
    test = False

    experiment_name = f'test'

    launcher = Launcher(experiment_name,
                        'experiment_config',
                        25,
                        memory=500,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=False,
                        conda_env='iprl'
                        )
    
    launcher = reproduce_lqr_experiments(launcher) # memory=500
    # launcher = reproduce_ship_steering(launcher) # memory=3000
    # launcher = reproduce_air_hockey(launcher) # memory=1000
    # launcher = reproduce_ball_stopping(launcher) # memory=3000

    print('name:', experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)