from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

	local = False
	test = False

	experiment_name = f'ship_all_best_25'
	
	launcher = Launcher(experiment_name,
						'el_ship_mi',
						25,
						memory=1000,
						days=2,
						hours=0,
						minutes=0,
						seconds=0,
						n_jobs=-1,
						use_timestamp=True)
	
	launcher.add_default_params(
								n_tilings=1,
								sigma_init=7e-2,
								n_epochs=50, fit_per_epoch=1, ep_per_fit=25,
								mi_type='regression', bins=4)

	# # all best
	# launcher.add_experiment(alg='ConstrainedREPS', eps=5.3, kappa=14.)
	# launcher.add_experiment(alg='REPS', eps=0.9)
	# launcher.add_experiment(alg='REPS_MI', eps=0.9, k=25, sample_type='percentage', gamma=0.6)
	# launcher.add_experiment(alg='ConstrainedREPSMI', eps=5.3, kappa=14., k=145, sample_type='percentage', gamma=0.8)
	# launcher.add_experiment(alg='RWR', eps=0.9)
	
	
	# eps and kappa:
	# for eps in np.arange(0.3, 8.0, 0.2):
	# 	eps = round(eps,1)
	# 	# REPS
	# 	launcher.add_experiment(alg='REPS', eps=eps)
	# 	# RWR
	# 	launcher.add_experiment(alg='RWR', eps=eps)

	# 	for kappa in np.arange(9., 20., 1.):
	# 		kappa = round(kappa,1)
	# 		# Constrained REPS
	# 		launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa)

	# # k -> ship has 150 parameters
	# # hyperparameters according to max reward
	# for k in range(5, 150, 10):
	# 	eps = 5.3
	# 	kappa = 14.0
	# 	launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type=None, eps=eps, kappa=kappa)
	# 	launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='importance', eps=eps, kappa=kappa)
	
	# 	for gama in np.arange(0.1, 1.0, 0.1):
	# 		gama = round(gama,1)
	# 		launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='percentage', gamma=gama, eps=eps, kappa=kappa)

	# 	eps = 0.9
	# 	launcher.add_experiment(alg='REPS_MI', k=k, sample_type='importance', eps=eps)

	# 	for gama in np.arange(0.1, 1.0, 0.1):
	# 		gama = round(gama,1)
	# 		launcher.add_experiment(alg='REPS_MI',k=k, sample_type='percentage', gamma=gama, eps=eps)

	# # MI vs Pearson
	# for sample_type in ['PRO', 'importance', 'percentage']:
	# 	for method in ['MI', 'Pearson']:
	# 		if sample_type == 'percentage':
	# 				launcher.add_experiment(alg='ConstrainedREPSMI', sample_type=sample_type, method=method, gamma=0.8, k=145, kappa=14.0, eps=5.3)
	# 				launcher.add_experiment(alg='REPS_MI', sample_type=sample_type, method=method, gamma=0.6, k=25, eps=0.9) 
	# 		else:
	# 			launcher.add_experiment(alg='ConstrainedREPSMI', sample_type=sample_type, method=method, k=145, kappa=14.0, eps=5.3)
	# 			launcher.add_experiment(alg='REPS_MI', sample_type=sample_type, method=method, k=25, eps=0.9)

	# print(experiment_name)

	launcher.run(local, test)
