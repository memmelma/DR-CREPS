# Dimensionality Reduction and Prioritized Exploration for Policy Search
by [Memmel M.](https://memmelma.github.io/), [Liu P.](https://www.ias.informatik.tu-darmstadt.de/Team/PuzeLiu), [Tateo D.](https://www.ias.informatik.tu-darmstadt.de/Team/DavideTateo), [Peters J.](https://www.ias.informatik.tu-darmstadt.de/Team/JanPeters)

[Abstract] _Black-box policy optimization, a class of reinforcement learning algorithms, explores and updates policies at the parameter level. These types of algorithms are applied widely in robotics applications with movement primitives or non-differentiable policies and are particularly relevant where exploration at the action level could cause actuator damage or other safety issues. However, they do not scale well with the increasing dimensionality of the policy, triggering high demand for samples, expensive to obtain on real-world systems. In most setups, policy parameters do not contribute equally to the return. Thus, identifying the parameters contributing most allows us to narrow down the exploration and speed up the learning. Furthermore, updating only the effective parameters requires fewer samples and thereby solves the scalability issue. We present a novel method to prioritize the exploration of effective parameters and cope with full covariance matrix updates. Our algorithm learns faster than recent approaches and requires fewer samples to achieve state-of-the-art results. To select the effective parameters, we consider both the Pearson correlation coefficient and the Mutual Information. We showcase the capabilities of our approach on the Relative Entropy Policy Search algorithm in several simulated environments, including robotics simulations._

![Algorithm Overview](https://github.com/memmelma/DR-CREPS/blob/main/algorithm_overview.png)

### Getting Started
Code tested with ```python==3.7```. We use [MushroomRL-v.1.7.0](https://github.com/MushroomRL/mushroom-rl/tree/a0eaa2cf8001e433419234a9fc48b64170e3f61c), [Benchmark-v2](https://github.com/MushroomRL/mushroom-rl-benchmark/tree/192ab521a693f5a210b851c87a1f7d31aedfaa2f), and [experiment_launcher](https://git.ias.informatik.tu-darmstadt.de/common/experiment_launcher.git). Required packages must be installed via ```pip3 install -r requirements.txt```.

An example to setup the repository using Anaconda:
```
conda create --name drcreps python=3.7
git pull https://github.com/memmelma/DR-CREPS.git
cd DR-CREPS
pip install -r requirements.txt
```

### Reproduce Results and Launch Custom Experiments
To reproduce our results simply execute ```python run.py``` which contains all experiments from our paper.
Note that if you do not run on a cluster that supports _SLURM_ or _Joblib_ you have to set ```local=True``` for a successful execution.
You can run custom experiments similar to the given examples. 

### Launch Single Experiment
To run a single (local) experiment directly execute ```experiment_config.py``` and specify the parameters via the command line or directly in the file by overriding the default parameter dictionary. The script corresponding to the given algorithm and environment setup will then be called from the ```experiments/``` directory. This mode does not support 
_SLURM_ or _Joblib_.

### Set Up Your Own Environment/Experiments
We provide implementations for the following environments:
- ```lqr```
- ```ship_steering```
- ```air_hockey```
- ```ball_stopping```
- ```optimization_function```: e.g. Himmelblau, Rosenbrock

To get an idea of how to setup more detailed experiments please have a look at ```experiments/ENVIRONMENT/``` where you can find scripts for the following algorithms:
- ```el_es.py```: natural evolution strategies (ES, NES)
- ```el_grad.py```: gradient based methods (REINFORCE, TRPO, PPO)
- ```el_optim.py```: classic optimizers (Nelder-Mead, L-BFGS-B)
- ```el_bbo.py```: policy search algorithms (RWR, REPS, CREPS, CEM, MORE, **DR-REPS**, **DR-CREPS**, PRO)

### Reuse Our Code!
Our implementation builds on [MushroomRL](https://github.com/MushroomRL/mushroom-rl). It also integrates nicely with it, i.e., all of our algorithms and environments either inherit directly from MushroomRL modules or implement the same intuitive interfaces.
