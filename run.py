from experiment_launcher import Launcher
from experiments import reproduce_diag_lqr_experiments, reproduce_lqr_experiments, reproduce_ship_steering, reproduce_air_hockey, reproduce_ball_stopping

if __name__ == '__main__':

    local = False
    test = False

    # LQR
    experiment_name = f'CR_LQR_diag'
    launcher = Launcher(experiment_name,
                        'experiment_config',
                        25,
                        memory=500,
                        use_timestamp=False,
                        conda_env='iprl'
                        )
    
    launcher = reproduce_diag_lqr_experiments(launcher, local)

    print('name:', experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)


    experiment_name = f'CR_LQR'
    launcher = Launcher(experiment_name,
                        'experiment_config',
                        25,
                        memory=500,
                        use_timestamp=False,
                        conda_env='iprl'
                        )
    
    launcher = reproduce_lqr_experiments(launcher, local)

    print('name:', experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)


    # ShipSteering
    experiment_name = f'CR_ShipSteering'
    launcher = Launcher(experiment_name,
                        'experiment_config',
                        25,
                        memory=3000,
                        use_timestamp=False,
                        conda_env='iprl'
                        )
    launcher = reproduce_ship_steering(launcher, local)

    print('name:', experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)


    # AirHockey
    experiment_name = f'CR_AirHockey'
    launcher = Launcher(experiment_name,
                        'experiment_config',
                        25,
                        memory=1000,
                        use_timestamp=False,
                        conda_env='iprl'
                        )
    launcher = reproduce_air_hockey(launcher, local)

    print('name:', experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)


    # BallStopping
    experiment_name = f'CR_BallStopping'
    launcher = Launcher(experiment_name,
                        'experiment_config',
                        25,
                        memory=3000,
                        use_timestamp=False,
                        conda_env='iprl'
                        )
    launcher = reproduce_ball_stopping(launcher, local)

    print('name:', experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)