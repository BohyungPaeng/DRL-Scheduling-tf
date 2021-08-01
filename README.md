# Introduction

Distributed version of Deep Reinforcement Learning (DRL)-based scheduling agents for minimizing tardiness

Codes are implemented by python 3.6, tensorflow 1.14

When using this repository for academic purposes,
please cites our paper "Deep Reinforcement Learning for Minimizing Tardiness in Parallel Machine Scheduling With Sequence Dependent Family Setups", IEEE Access (2021)

URL: https://ieeexplore.ieee.org/document/9486959

# Outline
agent package : replay buffer, DRL agent trainer

env package: simulator environments with parallel machines, Wrapper() class for generating 2-D states

model package: Deep Q-Network (DQN) and relevant methods modified from other researchers' previous works

utils package: core objects, logging experimental results, state visualization

config.py: specify experiments configuration including hyperparameters of DQN

main.py: train DQN and automatic validation (tensorboard)

test.py: performance comparison with heuristics, obtain schedules from trained DQNs


# Requirements
py36 tf14

pip install opencv-python pandas

# Instruction for general users
Run main.py, then experiments with Dataset1 (paper) can be reproduced.

Every hyperparameters and simulation configurations are specified in config.py (args)

Users might modify each argument. For example, to reproduce the experiment of Dataset7 (paper) with learning rate of 0.1, use commands as follow:

python main.py --did=4 --F=7 --lr=0.1

For the usage of arguments, check annotations of parser definition in config.py


Experimental results are automatically logged in new folder named 'results'

For distributed version, no files are made in best_models, gantt 

At every args.save_freq episodes, trained DQN are saved as checkpoint file in 'results/models' folder.

Simultaneously, tensorboard summary are reported including following information:

1. training loss (per each train step), cumulative rewards (per each episode), cumulative Q-values (per each episode)
2. validation results (saved DQN) as cumulative rewards, which are equal to total tardiness in this research.

# Instruction for reproducing the experiments
8 Datasets can be reproduced by refering annotations in config.py

Stochastic processing and setup time (table 5 in the paper) can be reproduced by modifying STOCHASTICITY variable in env/simul_pms.py.

For datasets 1 to 4, I recommend args.bucket=7200, args.save_freq=1000

For datasets 5 to 8, args.bucket=5400, args.save_freq=20

Other hyperparameters does not need to be modified for reproducing the results.

To exclude parameter sharing (FBS-1D in the paper), change args.state_type=1D (default is 2D)

For LBF-Q, args.oopt=upm2007

For TPDQN, args.oopt=fab2018

For heuristics including SSTEDD, COVERT, Run test.py --test_mode=logic --ropt=tardiness, then check the summary/test_performance_logic.csv (unit is reward, not tardiness hour)

# To advanced users, future researchers
As stated in the paper, DRL hyperparemeters are essential but hard to be optimized. 

Try various values of args.GAMMA, freq_tar, freq_on, warmup, eps, lr, and so on.

As in 150th line of main.py, new random seeds are setted at every new episode. This scheme can be re-considered.


After being acquiant to the codes, modify util_sim.py, simul_pms.py to simulate scheduling problems of yours own. 

(Distribution of production requirements, due-dates, initial machine status, Time table of processing, setup time, ...)

Modify wrapper.py to generate state vectors appropriate for your problems.

