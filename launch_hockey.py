from itertools import product
from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

	local = False
	test = False

	# experiment_name = f'ball_sample'
	experiment_name = f'hockey_eps_kappa'
	
	launcher = Launcher(experiment_name,
						'el_air_hockey_mi',
						10,
						memory=1000,
						days=2,
						hours=24,
						minutes=0,
						seconds=0,
						use_timestamp=False)
	
	launcher.add_default_params(n_epochs=750,
                                fit_per_epoch=1, ep_per_fit=25,
								n_basis=20, horizon=120,
								sigma_init=1e-0, distribution='diag', bins=4, mi_type='regression', mi_avg=0)
	
	# # all best
	# launcher.add_experiment(alg='ConstrainedREPS', eps=2.7, kappa=6.)
	# launcher.add_experiment(alg='REPS', eps=0.7)
	# launcher.add_experiment(alg='REPS_MI', eps=0.7, k=110, sample_type='percentage', gamma=0.9)
	# launcher.add_experiment(alg='ConstrainedREPSMI', eps=2.7, kappa=6., k=65, sample_type='percentage', gamma=0.5)
	# launcher.add_experiment(alg='RWR', eps=1.7)
	

	# eps and kappa:
	for eps in np.arange(0.2, 3.3, 0.5):
		eps = round(eps,1)
		# REPS
		launcher.add_experiment(alg='REPS', eps=eps)
		# RWR
		launcher.add_experiment(alg='RWR', eps=eps)
		
		for kappa in np.arange(2., 12., 2.):
			kappa = round(kappa,1)
			# Constrained REPS
			launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa)

	# # hyperparameters according to max reward
	# for k in range(5, 140, 15):
	# 	eps = 2.7
	# 	kappa = 6.0
	# 	launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type=None, eps=eps, kappa=kappa)
	# 	launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='importance', eps=eps, kappa=kappa)
	# 	for gama in np.arange(0.1, 1.0, 0.2):
	# 		gama = round(gama,1)
	# 		launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='percentage', gamma=gama, eps=eps, kappa=kappa)

	# 	eps = 0.7
	# 	# launcher.add_experiment(alg='REPS_MI', k=k, sample_type=None, eps=eps) # -> REPS
	# 	launcher.add_experiment(alg='REPS_MI', k=k, sample_type='importance', eps=eps)

	# 	for gama in np.arange(0.1, 1.0, 0.2):
	# 		gama = round(gama,1)
	# 		launcher.add_experiment(alg='REPS_MI', k=k, sample_type='percentage', gamma=gama, eps=eps)

	# # MI vs Pearson
	# launcher.add_experiment(alg='REPS_MI', k=110, method='Pearson', sample_type='percentage', gamma=0.9, eps=0.7)
	# launcher.add_experiment(alg='REPS_MI', k=110, method='Pearson', sample_type='PRO', eps=0.7)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=65, method='Pearson', sample_type='percentage', gamma=0.5, eps=2.7, kappa=6.)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=65, method='Pearson', sample_type='PRO', eps=2.7, kappa=6.)
	
	# launcher.add_experiment(alg='REPS_MI', k=110, method='MI', sample_type='percentage', gamma=0.9, eps=0.7)
	# launcher.add_experiment(alg='REPS_MI', k=110, method='MI', sample_type='PRO', eps=0.7)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=65, method='MI', sample_type='percentage', gamma=0.5, eps=2.7, kappa=6.)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=65, method='MI', sample_type='PRO', eps=2.7, kappa=6.)

	print(experiment_name)
	print('experiments:', len(launcher._experiment_list))

	launcher.run(local, test)
