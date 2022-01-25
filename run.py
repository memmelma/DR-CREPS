from experiment_launcher import Launcher
from experiments import reproduce_lqr_experiments, reproduce_ship_steering, reproduce_air_hockey, reproduce_ball_stopping


if __name__ == '__main__':

    local = False
    test = False

    # LQR
    experiment_name = f'LQR_camera_ready'
    launcher = Launcher(experiment_name,
                        'experiment_config',
                        25,
                        memory=500,
                        days=2,
                        use_timestamp=False,
                        conda_env='iprl'
                        )
    
    launcher = reproduce_lqr_experiments(launcher)
    
    print('name:', experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)


    # ShipSteering
    experiment_name = f'BallStopping_camera_ready'
    launcher = Launcher(experiment_name,
                        'experiment_config',
                        25,
                        memory=3000,
                        days=2,
                        use_timestamp=False,
                        conda_env='iprl'
                        )
    launcher = reproduce_ship_steering(launcher)

    print('name:', experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)


    # AirHockey
    experiment_name = f'AirHockey_camera_ready'
    launcher = Launcher(experiment_name,
                        'experiment_config',
                        25,
                        memory=1000,
                        days=2,
                        use_timestamp=False,
                        conda_env='iprl'
                        )
    launcher = reproduce_air_hockey(launcher)

    print('name:', experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)


    # BallStopping
    experiment_name = f'BallStopping_camera_ready'
    launcher = Launcher(experiment_name,
                        'experiment_config',
                        25,
                        memory=3000,
                        days=2,
                        use_timestamp=False,
                        conda_env='iprl'
                        )
    launcher = reproduce_ball_stopping(launcher)

    print('name:', experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)