"""
Plot experiment comparing Flex-VFL, Sync-VFL, and Async-VFL
"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import random
import os
import glob

font = {'family' : 'DejaVu Sans',
#        'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)
colors=['#6aa2fc', '#fc8181', '#a5ff9e', '#3639ff', '#ff3636', '#13ba00', '#ff62f3']

def all_seeds(prefix, time_per_epoch):
    print(prefix)
    files = glob.glob(f'{prefix}_lrmain0.001_lrbert5e-05_decay0.0001_seed*.pkl')
    pickles = []    
    min_len = 99999
    for f in files:
        pkl = pickle.load(open(f, 'rb'))
        pkl = np.array([float(val['mae']) for val in pkl])
        if 'pbcd' in prefix and len(pkl) <= 10:
            continue
        if '50' in prefix and len(pkl) <= 1000:
            continue
        pickles.append(pkl)
        min_len = min(min_len, len(pkl))
    for i in range(len(pickles)):
        pickles[i] = pickles[i][:min_len]
    pickles = np.array(pickles)

    time_to_reach = []
    for i in range(len(pickles)):
        final_epoch = 0
        time_so_far = 0
        while pickles[i][final_epoch] > 0.65:
            time_so_far += time_per_epoch
            final_epoch += 1
            if final_epoch >= len(pickles[i]):
                time_so_far = float('inf')
                break
        time_to_reach.append(time_so_far/1000)
    time_2 = []
    for i in range(len(time_to_reach)):
        if time_to_reach[i] != float('inf'):
            time_2.append(time_to_reach[i])
    if len(time_2) > 0:
        time_to_reach = time_2
    time_to_reach = np.array(time_to_reach)
    min_ind, max_ind = np.argmin(time_to_reach), np.argmax(time_to_reach)
    time_avg = np.average(time_to_reach)
    time_std = np.std(time_to_reach)
    print(f'& {time_avg:.2f} $\pm$ {time_std:.2f}')
    
    avg = np.average(pickles, axis=0)
    std = np.std(pickles, axis=0)

    return (avg, std)

num_batches = 326
epochs = 2000
local_time = 1
Q = 20
slowest = 20/5

# For each amount of server round-trip communication
comm_times = [1.0, 10.0, 50.0]
for comm_time in comm_times:
    # Parse results
    accs3, std3 = all_seeds(f"results_algpbcd_servertime_1.0_datasetmosei", (comm_time+local_time*slowest)*num_batches)
    accs2, std2 = all_seeds(f"results_algvafl_servertime_{comm_time}_datasetmosei", num_batches)
    accs4, std4 = all_seeds(f"results_algsync1_servertime_1.0_datasetmosei", (comm_time+local_time*Q)*num_batches)
    accs5, std5 = all_seeds(f"results_algsync_servertime_1.0_datasetmosei", (comm_time+local_time*Q*slowest)*num_batches)
    accs1, std1 = all_seeds(f"results_algflex_servertime_1.0_datasetmosei", (comm_time+local_time*Q)*num_batches) 
