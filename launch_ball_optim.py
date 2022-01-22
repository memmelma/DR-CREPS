from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

	local = False
	test = False

	experiment_name = f'ball_optim'

	launcher = Launcher(experiment_name,
						'el_ball_optim',
						25,
						memory=3000,
						days=2,
						hours=0,
						minutes=0,
						seconds=0,
						conda_env='iprl',
						use_timestamp=False)
	
	launcher.add_default_params(n_basis=20, horizon=750, \
								ep_per_fit=1, fit_per_epoch=1, \
								sigma_init=1e-0)

	n_samples = 7000

	n_epochs = n_samples
	launcher.add_experiment(alg='NM', n_epochs=n_epochs)
	launcher.add_experiment(alg='BFGS', n_epochs=n_epochs)

	print(experiment_name)
	print('experiments:', len(launcher._experiment_list))
	launcher.run(local, test)

	