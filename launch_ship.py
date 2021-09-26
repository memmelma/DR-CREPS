from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

	local = False
	test = False

	# experiment_name = f'ship_3_tiles_full_besty_10'
	experiment_name = f'ship_3_tiles_full_sanity'
	
	launcher = Launcher(experiment_name,
						'el_ship_mi',
						10,
						memory=1000,
						days=2,
						hours=0,
						minutes=0,
						seconds=0,
						# n_jobs=-1,
						use_timestamp=False)
	
	# launcher.add_default_params(
	# 							n_tilings=3,
	# 							sigma_init=7e-2,
	# 							n_epochs=75, fit_per_epoch=1, ep_per_fit=25,
	# 							mi_type='regression', bins=4)

	# # all best
	# launcher.add_experiment(alg='ConstrainedREPS', eps=5.3, kappa=14.)
	# launcher.add_experiment(alg='REPS', eps=0.9)
	# launcher.add_experiment(alg='REPS_MI', eps=0.9, k=25, method='MI', sample_type='percentage', gamma=0.6)
	# launcher.add_experiment(alg='ConstrainedREPSMI', eps=5.3, kappa=14., k=145, method='MI',  sample_type='percentage', gamma=0.8)
	# launcher.add_experiment(alg='RWR', eps=0.9)
	
	
	# eps and kappa:
	# for eps in np.arange(0.3, 8.0, 0.2):
	# for eps in np.arange(0.1, 3.0, 0.2):
	# 	eps = round(eps,1)
	# 	# REPS
	# 	launcher.add_experiment(alg='REPS', eps=eps)
	# 	# # RWR
	# 	launcher.add_experiment(alg='RWR', eps=eps)
	# for eps in np.arange(6.5, 14.0, 0.5):
	# 	eps = round(eps,1)
	# 	for kappa in np.arange(6., 14., 1.):
	# 		kappa = round(kappa,1)
	# 		# Constrained REPS
	# 		launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa)

	# # k -> ship has 150 parameters
	# # hyperparameters according to max reward
	# for k in range(5, 150, 10):
	# 	eps = 9.5
	# 	kappa = 12.0

	# 	# launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type=None, eps=eps, kappa=kappa)
	# 	# launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='importance', eps=eps, kappa=kappa)
	
	# 	for gama in np.arange(0.1, 1.0, 0.1):
	# 		gama = round(gama,1)
	# 		launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='percentage', gamma=gama, eps=eps, kappa=kappa)

	# 	eps = 1.1
	# 	# launcher.add_experiment(alg='REPS_MI', k=k, sample_type='importance', eps=eps)

	# 	for gama in np.arange(0.1, 1.0, 0.1):
	# 		gama = round(gama,1)
	# 		launcher.add_experiment(alg='REPS_MI',k=k, sample_type='percentage', gamma=gama, eps=eps)

	### MISSING
	
	## ORACLE
	# for eps in np.arange(8., 11., 0.5):
	# 	eps = np.round(eps, 1)
	# 	for kappa in np.arange(12., 16., 1.):
	# 		kappa = np.round(kappa,1)
	# 		for gama in np.arange(0.1, 1.0, 0.1):
	# 			launcher.add_experiment(alg='ConstrainedREPSMIOracle', sample_type='percentage', gamma=gama, eps=eps, kappa=kappa)

	# for eps in np.arange(0.9, 2., 0.2):
	# 	eps = np.round(eps, 1)
	# 	for gama in np.arange(0.1, 1.0, 0.1):
	# 		gama = round(gama,1)
	# 		launcher.add_experiment(alg='REPS_MI_ORACLE', sample_type='percentage', gamma=gama, eps=eps)

	# launcher.add_experiment(alg='ConstrainedREPSMIOracle', sample_type='percentage', gamma=0.9, eps=9.5, kappa=12.)
	# launcher.add_experiment(alg='REPS_MI_ORACLE', sample_type='percentage', gamma=0.9, eps=1.1)

	# # MI vs Pearson
	# for sample_type in ['PRO', 'importance', 'percentage']:
	# 	for method in ['MI', 'Pearson']:
	# 		if sample_type == 'percentage':
	# 				launcher.add_experiment(alg='ConstrainedREPSMI', sample_type=sample_type, method=method, gamma=0.8, k=145, kappa=14.0, eps=5.3)
	# 				launcher.add_experiment(alg='REPS_MI', sample_type=sample_type, method=method, gamma=0.6, k=25, eps=0.9) 
	# 		else:
	# 			launcher.add_experiment(alg='ConstrainedREPSMI', sample_type=sample_type, method=method, k=145, kappa=14.0, eps=5.3)
	# 			launcher.add_experiment(alg='REPS_MI', sample_type=sample_type, method=method, k=25, eps=0.9)

	### SHIP FULL COVARIANCE ######################################################################################################################################################

	launcher.add_default_params(
								n_tilings=3,
								sigma_init=7e-2,
								fit_per_epoch=1,
								mi_type='regression', bins=4)

	n_samples = 3500

	# # Cholesky Gaussian Distribution
	distribution = 'cholesky'

	# for ep_per_fit in [15, 30]:
	ep_per_fit = 15
	n_epochs = n_samples // ep_per_fit

	# 	# REPS & RWR
	# 	# for eps in np.arange(0.1, 2.5, 0.3):
	# 	# 	eps = np.round(eps, 1)
	# 	# 	launcher.add_experiment(alg='REPS', distribution=distribution, eps=eps, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
	# 	# 	launcher.add_experiment(alg='RWR', distribution=distribution, eps=eps, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

	# 	# Constrained REPS
	# 	for eps in np.arange(1.9, 3.9, 0.5):
	# 		eps = np.round(eps, 1)
	# 		for kappa in np.arange(17., 30., 3.):
	# 			kappa = np.round(kappa, 1)
	# 			launcher.add_experiment(alg='ConstrainedREPS', distribution=distribution, eps=eps, kappa=kappa, n_epochs=n_epochs, ep_per_fit=ep_per_fit)


	# # Constrained REPS and MORE
	# for eps in np.arange(2.4, 5.4, 1.0):
		# kappa = 20.

		# launcher.add_experiment(alg='ConstrainedREPS', distribution=distribution, eps=eps, kappa=kappa, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
		# launcher.add_experiment(alg='MORE', distribution=distribution, eps=eps, kappa=kappa, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

	# Cholesky Gaussian Distribution for MI based algorithms
	for eps in [2.4, 3.4]:
		kappa = 20
		ep_per_fit = 15
		n_epochs = n_samples // ep_per_fit
		launcher.add_experiment(alg='ConstrainedREPSMI', distribution='cholesky', eps=eps, kappa=kappa, sample_type=None, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
		k = 15
		launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=0.1, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
		launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=0.5, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
		launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=0.9, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)


	# # Constrained REPS with MI
	# eps = 3.4
	# kappa = 20.
	# for k in np.arange(100, 500, 100):
	# 	for gama in np.arange(0.1, 1., 0.2):
	# 		gama = np.round(gama,1)
	# 		launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution=distribution, eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=gama, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)

	# for eps in [1.1]:
	# 	for k in np.arange(10, 50, 10):
	# 		for ep_per_fit in [int(1.5* k), int(2 * k), int(2.5 * k)]:
	# 			n_epochs = n_samples // ep_per_fit
	# 			# for gama in np.arange(0.1, 1., 0.2):
	# 			for gama in [0.1, 0.5, 0.9]:
	# 				gama = np.round(gama,1)
	# 				launcher.add_experiment(alg='REPS_MI_full', distribution=distribution, eps=eps, k=k, sample_type='percentage', gamma=gama, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)

	# # Constrained REPS MI
	# for k in np.arange(15, 450, 25):
	
	# for eps in np.arange(2.4, 3.5, 0.5):
	# 	# for k in np.arange(5, 30, 15):
	# 	for k in np.arange(25, 450, 100):
	# 		k = int(k)
	# 		ep_per_fit = 15
	# 		n_epochs = n_samples // ep_per_fit
			
	# 		for gama in np.arange(0.1, 1., 0.2):
	# 			gama = np.round(gama,1)
	# 			launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution=distribution, eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=gama, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)

	# # Best eps from REPS
	# eps = 0.5
	
	# # REPS MI
	# for k in np.arange(15, 450, 25):
	# 	 k = int(k)
	# 	 # n_samples required for full covariance / n_parameters = 2.5
	# 	 # ep_per_fit = int(k * 3)
	# 	 for ep_per_fit in [k*3, k*4]:
	# 		 n_epochs = n_samples // ep_per_fit
		
	# 		 for gama in np.arange(0.1,1.0,0.1):
	# 			 gama = np.round(gama,1)
	# 			 launcher.add_experiment(alg='REPS_MI_full', distribution=distribution, eps=eps, k=k, sample_type='percentage', gamma=gama, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
	# 	 # launcher.add_experiment(alg='REPS_MI_full', distribution=distribution, eps=eps, k=k, sample_type=None, method='MI', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
	
	
	# for 3 tilings, k goes to 450 !!!
	print(experiment_name)
	print('experiments:', len(launcher._experiment_list))
	launcher.run(local, test)
