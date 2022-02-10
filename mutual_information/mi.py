import os
import time
import numpy as np
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import analytical_MI, compute_MI, get_mean_and_confidence, get_style

def run(args):
	m, n, s, bins, n_runs, random_seed = args
	mi_trial = []
	for i in range(n_runs):
		x, y, I_gt = analytical_MI(m=m, n=n, samples=s, random_seed=random_seed+i)
		mi_trial += [compute_MI(x, y, I_gt, bins, random_seed=random_seed+i)]
	return mi_trial
	
def plot_samples(m, n, samples=[5, 10, 15], bins=[3, 4], n_runs=25, y_axis=None, log_dir='', random_seed=42, pool=False, override=False, print_legend=False, save_legend=False, pdf=False):
	
	file_name = f'mi_m_{m}_n_{n}_bins_{bins}'
	mi_runs = []

	if os.path.isfile(os.path.join(log_dir, 'data', file_name+'.npy')) and not override: 
		mi_runs, legend, colors, linestyle = np.load(os.path.join(log_dir, 'data', file_name+'.npy'), allow_pickle=True)
		print('Open existing file')
	else:
		start = time.time()
		if pool:
			processes = min(len(samples),multiprocessing.cpu_count())
			print(f'Multiprocessing with {processes} processes')
			
			with multiprocessing.Pool(processes=processes) as pool:
				tasks = [ [m,n,s,bins,n_runs,random_seed] for s in samples ]
				mi_runs = list(tqdm(pool.imap(run, tasks), total=len(samples)))
		else:
			for s in tqdm(samples):
				mi_trial = run([m,n,s,bins,n_runs,random_seed])
				mi_runs += [mi_trial]

		print(f'Took {np.round(time.time() - start,6)} seconds')
		legend, colors, linestyle = get_style()
		np.save(open(os.path.join(log_dir, 'data', file_name+'.npy'), 'wb'), (mi_runs, legend, colors, linestyle))

	for k, mi_runs in enumerate(np.transpose(np.array(mi_runs, dtype=float), (1,0,2))):
		
		mi_runs = np.expand_dims(mi_runs, 0)
		mean, ci = get_mean_and_confidence(mi_runs)

		fig, ax = plt.subplots()
		for i in range(0, mean.shape[1]-1):
			ax.plot(samples, mean[:,1:][:,i],
					color=colors[1:][i//len(bins)], 
					linestyle=linestyle[i%len(bins)])
			ax.fill_between(samples, mean[:,1:][:,i]+ci[0][:,1:][:,i], mean[:,1:][:,i]+ci[1][:,1:][:,i], 
							color=colors[1:][i//len(bins)], alpha=.2)
		ax.plot(samples,  mean[:,0], color=colors[0], linestyle='-')

		legend_elements = []
		for color, leg in zip(colors, legend):
			legend_elements += [Line2D([0], [0], color=color, lw=1, label=leg)]
		for j, bin in enumerate(bins):
			legend_elements += [Line2D([0], [0], color='black', lw=1, label=f'bins/$k$={bin}', linestyle=linestyle[j])]
		
		if print_legend:
			ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2),
				fancybox=True, ncol=4)
		elif save_legend:
			legend = ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=10)

			def export_legend(legend, filename_legend="legend_mi_apx", expand=[-5,-5,5,5]):
				filename_legend = os.path.join(log_dir, filename_legend)
				fig  = legend.figure
				fig.canvas.draw()
				bbox  = legend.get_window_extent()
				bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
				bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
				if pdf:
					fig.savefig(filename_legend+'.pdf', dpi="figure", bbox_inches=bbox)
				fig.savefig(filename_legend+'.png', dpi="figure", bbox_inches=bbox)
				print(f'Exported legend to {filename_legend}')
			
			return export_legend(legend)
			
		plt.subplots_adjust(bottom=0.25)

		x_0, x_1 = samples[0], samples[-1]
		plt.xlim(x_0, x_1)
		if y_axis is not None:
			y_0, y_1 = y_axis[0], y_axis[1]
			plt.ylim(y_0, y_1)

		ax.set_xlabel('samples', fontsize=20)
		ax.set_ylabel('$MI[X;Y]$', fontsize=20)

		plt.tight_layout()
		plt.grid()

		if pdf:
			plt.savefig(os.path.join(log_dir, 'imgs', file_name+f'_dim_{k}.pdf'))
		plt.savefig(os.path.join(log_dir, 'imgs', file_name+f'_dim_{k}.png'))

if __name__ == '__main__':

	log_dir = 'logs_mi'
	img_dir = os.path.join(log_dir, 'imgs')
	data_dir = os.path.join(log_dir, 'data')
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(img_dir, exist_ok=True)
	os.makedirs(data_dir, exist_ok=True)

	plot_samples(m=1, n=1, samples=np.arange(5, 500, 5), bins=[4], n_runs=4, y_axis=[-0.1,0.6], log_dir=log_dir, pool=True, override=True, print_legend=False)

	