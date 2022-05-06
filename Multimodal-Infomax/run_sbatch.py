import numpy as np
import os
import random

seeds = [707412115,1928644128,16910772,1263880818,1445547577]
for seed in seeds:
    os.system(f'./run_slurm.sh {seed} flex 1.0')
    os.system(f'./run_slurm.sh {seed} sync 1.0')
    os.system(f'./run_slurm.sh {seed} sync1 1.0')
    os.system(f'./run_slurm.sh {seed} pbcd 1.0')
    server_times = [1.0, 10.0, 50.0]
    for server_time in server_times:
        os.system(f'./run_slurm.sh {seed} vafl {server_time}')
    os.system(f'./run_adapt.sh {seed} adapt 1.0')
    os.system(f'./run_adapt.sh {seed} nonadapt 1.0')

