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
        'size'   : 15}
plt.rc('font', **font)
colors=['#6aa2fc', '#fc8181', '#a5ff9e', '#3639ff', '#ff3636', '#13ba00', '#ff62f3']

def smooth(arr, std):
    first_val = -1
    first_std = -1
    val_ind = -1
    for i in range(len(arr)):
        arr[i] = np.max(arr[:i+1])
    for i in range(len(arr)):
        if arr[i] != first_val:
            if val_ind != -1:
                smoothed_vals = np.arange(first_val, arr[i], (arr[i]-first_val)/(i-val_ind))
                smoothed_stds = np.arange(first_std, std[i], (std[i]-first_std)/(i-val_ind))
                if len(smoothed_vals) != len(arr[val_ind:i]):
                    smoothed_vals = smoothed_vals[:len(smoothed_vals)-1]
                if len(smoothed_stds) != len(std[val_ind:i]):
                    smoothed_stds = smoothed_stds[:len(smoothed_stds)-1]
                arr[val_ind:i] = smoothed_vals
                std[val_ind:i] = smoothed_stds 
            first_val = arr[i]
            first_std = std[i]
            val_ind = i
    return np.array(arr), np.array(std)

def all_seeds(prefix, time_per_epoch):
    print(prefix)
    files = glob.glob(f'{prefix}_lrmain0.001_lrbert5e-05_decay0.0001_seed*.pkl')
    pickles = []    
    min_len = 99999
    for f in files:
        pkl = pickle.load(open(f, 'rb'))
        pkl = np.array([float(val['mae']) for val in pkl])
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
    print(avg)
    std = np.std(pickles, axis=0)

    return (avg, std)

num_batches = 326
epochs = 2000
local_time = 1
Q = 20
comm_time = 1.0

# For each amount of server round-trip communication
# Parse results
accs1, std1 = all_seeds(f"results_algadapt_servertime_1.0_datasetmosei", (comm_time+local_time*Q)*num_batches) 
accs2, std2 = all_seeds(f"results_algnonadapt_servertime_1.0_datasetmosei", (comm_time+local_time*Q)*num_batches) 

# Plot loss
time = np.arange(0, (epochs+2)*(comm_time+local_time*Q)*num_batches, (comm_time+local_time*Q)*num_batches)
# Plot accuracy
fig, ax = plt.subplots()

accs2 = np.array(accs2, dtype=float)
std2 = np.array(std2, dtype=float)
accs1new, std1new = accs1, std1#smooth(accs1, std1)
accs2new, std2new = accs2, std2#smooth(accs2, std2)
plt.plot(time[:len(accs1new)], accs1new, color=colors[3], label='Adaptive Flex-VFL')
plt.plot(time[:len(accs1new)], accs2new[:len(accs1new)], color=colors[4], label='Flex-VFL')
ax.ticklabel_format(axis='x', style='sci', scilimits=(3,3))

plt.fill_between(np.linspace(0,time[len(accs1new)-1],len(accs1new)), accs1new - std1new, accs1new + std1new, alpha=0.3)
plt.fill_between(np.linspace(0,time[len(accs1new)-1],len(accs1new)), accs2new[:len(accs1new)] - std2new[:len(accs1new)], accs2new[:len(accs1new)] + std2new[:len(accs1new)], alpha=0.3)

plt.xlabel('Time Units')
plt.ylabel('MAE')

plt.legend(loc='upper right')
plt.ylim(0.6, 0.9)

plt.tight_layout()
plt.savefig(f'acc_Multimodal_adapt.png')
