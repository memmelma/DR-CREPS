from itertools import product
from experiment_launcher import Launcher

if __name__ == '__main__':

	local = False
	test = False

	experiment_name = f'ship_1_tile'
	
	launcher = Launcher(exp_name=experiment_name,
						python_file='ship_mi_el',
						n_exp=5,
						memory=6000,
						days=2,
						hours=0,
						minutes=0,
						seconds=0,
						n_jobs=-1,
						use_timestamp=True)
	
	launcher.add_default_params(eps=1., kappa=2,
                                n_tilings=1,
                                sigma_init=4e-1,
                                n_epochs=30, 
                                fit_per_epoch=1, ep_per_fit=25,   
                                mi_type='regression', bins=4, 
                                sample_type='percentage', gamma=0.1)

	launcher.add_experiment(alg='REPS')
	launcher.add_experiment(alg='REPS_MI', k=25)
	launcher.add_experiment(alg='REPS_MI', k=500)
	launcher.add_experiment(alg='REPS_MI', k=1000)
	launcher.add_experiment(alg='RWR')
	launcher.add_experiment(alg='ConstrainedREPS')
	launcher.add_experiment(alg='ConstrainedREPSMI', k=25)
	launcher.add_experiment(alg='ConstrainedREPSMI', k=500)
	launcher.add_experiment(alg='ConstrainedREPSMI', k=1000)

	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=3, k=0.2)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=3, k=0.4)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=3, k=0.6)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=4, k=0.8)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=3, k=25)

	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=4, k=0.2)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=4, k=0.4)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=4, k=0.6)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=4, k=0.8)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=4, k=25)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=5, eps=0.7, sample_type='percentage', gamma=0.1)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25, eps=0.7, sample_type='percentage', gamma=0.1)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=0.2, eps=0.7, sample_type='percentage', gamma=0.1)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=0.6, eps=0.7, sample_type='percentage', gamma=0.1)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=0.8, eps=0.7, sample_type='percentage', gamma=0.1)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=5, eps=0.7, sample_type='percentage', gamma=0.1, bins=3, mi_type='sample')
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25, eps=0.7, sample_type='percentage', gamma=0.1, bins=3, mi_type='sample')
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=5, eps=0.7, sample_type='percentage', gamma=0.1, bins=4, mi_type='sample')
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25, eps=0.7, sample_type='percentage', gamma=0.1, bins=4, mi_type='sample')

	# launcher.add_experiment(alg='REPS', sigma_init=1e-3)
	# launcher.add_experiment(alg='ConstrainedREPS', sigma_init=1e-1)
	# launcher.add_experiment(alg='ConstrainedREPSMI', sigma_init=1e-1)
	
	# launcher.add_experiment(alg='REPS', sigma_init=1e-1)
	# launcher.add_experiment(alg='MORE', sigma_init=1e-1)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='fixed', gamma=1e-5)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='percentage', gamma=0.1)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type='fixed', gamma=1e-5)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type='percentage', gamma=0.1)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=50)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=75)


	print(experiment_name)

	launcher.run(local, test)
