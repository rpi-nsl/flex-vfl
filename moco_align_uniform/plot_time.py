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

def all_seeds(prefix, mode, time_per_epoch):
    files = None
    if mode == "flex":
        files = glob.glob(f'results/{prefix}*_e15,30,50,200/test_acc5.pkl')
    elif mode == "sync":
        files = glob.glob(f'results/{prefix}*_e15,30,200/test_acc5.pkl')
    elif mode == "sync1":
        files = glob.glob(f'results/{prefix}*_e30,60,90,200/test_acc5.pkl')
    elif mode == "pbcd":
        files = glob.glob(f'results/{prefix}*_e150,300,500,200/test_acc5.pkl')
    else:
        files = glob.glob(f'results/{prefix}*_e150,300,500,200/test_acc5.pkl')
    print(f'results/{prefix}.pkl')
    pickles = []    
    min_len = 99999
    for f in files:
        pkl = pickle.load(open(f, 'rb'))
        if len(pkl) < 10:
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
        while pickles[i][final_epoch] < 60:
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
    #min_ind, max_ind = np.argmin(time_to_reach), np.argmax(time_to_reach)
    #time_to_reach = np.delete(time_to_reach, [min_ind, max_ind])
    time_avg = np.average(time_to_reach)
    time_std = np.std(time_to_reach)
    print(f'& {time_avg:.2f} $\pm$ {time_std:.2f}')
    
    avg = np.average(pickles, axis=0)
    std = np.std(pickles, axis=0)

    return (avg, std)

num_batches = 495 
epochs = 2000
local_time = 1
Q = 10
slowest = 10/2
Q1 = 2

# For each amount of server round-trip communication
comm_times = [1.0, 10.0, 50.0]
for comm_time in comm_times:

    # Parse results
    accs3, std3 = all_seeds("b256_lr0.03_modepbcd_st1.0_seed", "pbcd", (comm_time+local_time*slowest)*num_batches)   
    accs2, std2 = all_seeds(f"b256_lr0.03_modevafl_st{comm_time}_seed", "vafl", num_batches) 
    accs4, std4 = all_seeds("b256_lr0.03_modesync1_st1.0_seed", "sync1", (comm_time+local_time*Q)*num_batches) 
    accs5, std5 = all_seeds("b256_lr0.03_modesync_st1.0_seed", "sync", (comm_time+local_time*Q*slowest)*num_batches) 
    accs1, std1 = all_seeds("b256_lr0.03_modeflex_st1.0_seed", "flex", (comm_time+local_time*Q)*num_batches)         
    #accs2 = np.zeros((2000))
    #std2 = np.zeros((2000))

    # Convert x-axis to time in milliseconds
    VAFL_time = num_batches 
    # Flex-VFL 
    Our_ratio = VAFL_time/((comm_time+local_time*Q)*num_batches)
    # Sync-VFL
    Sync_ratio = VAFL_time/((comm_time+local_time*Q*slowest)*num_batches)
    # P-BCD
    Q1_ratio = VAFL_time/((comm_time+local_time*slowest)*num_batches)

    # Stretch Flex-VFL results to be on same time scale as VAFL
    r = 0
    j = 0
    accs1new = []
    std1new = []
    for i in range(len(accs2)):
        if j >= len(accs1):
            break
        accs1new.append(float(accs1[j]))
        std1new.append(float(std1[j]))
        r += Our_ratio 
        if r >= 1:
            r -= 1
            j += 1

    # Stretch P-BCD results to be on same time scale as VAFL
    r = 0
    j = 0
    accs3new = []
    std3new = []
    for i in range(len(accs2)):
        if j >= len(accs3):
            break
        accs3new.append(float(accs3[j]))
        std3new.append(float(std3[j]))
        r += Q1_ratio 
        if r >= 1:
            r -= 1
            j += 1

    # Stretch Sync-VFL results to be on same time scale as VAFL
    r = 0
    j = 0
    accs4new = []
    std4new = []
    for i in range(len(accs2)):
        if j >= len(accs4):
            break
        accs4new.append(float(accs4[j]))
        std4new.append(float(std4[j]))
        r += Sync_ratio 
        if r >= 1:
            r -= 1
            j += 1

    # Plot loss
    time = np.arange(0, (epochs+2)*num_batches, num_batches)
    # Plot accuracy
    fig, ax = plt.subplots()

    accs2 = np.array(accs2, dtype=float)
    std2 = np.array(std2, dtype=float)
    accs1new, std1new = smooth(accs1new, std1new)
    accs2new, std2new = smooth(accs2, std2)
    accs3new, std3new = smooth(accs3new, std3new)
    accs4new, std4new = smooth(accs4new, std4new)
    plt.plot(time[:len(accs1new)], accs1new, color=colors[3], label='Flex-VFL')
    plt.plot(time[:len(accs2new)], accs2new, color=colors[4], label='VAFL')
    plt.plot(time[:len(accs4new)], accs4new, color=colors[5], label='Sync-VFL')
    plt.plot(time[:len(accs3new)], accs3new, color=colors[6], label='P-BCD')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(3,3))

    plt.fill_between(np.linspace(0,time[len(accs1new)-1],len(accs1new)), accs1new - std1new, accs1new + std1new, alpha=0.3)
    plt.fill_between(np.linspace(0,time[len(accs2new)-1],len(accs2new)), accs2new - std2new, accs2new + std2new, alpha=0.3)
    plt.fill_between(np.linspace(0,time[len(accs4new)-1],len(accs4new)), accs4new - std4new, accs4new + std4new, alpha=0.3)
    plt.fill_between(np.linspace(0,time[len(accs3new)-1],len(accs3new)), accs3new - std3new, accs3new + std3new, alpha=0.3)

    plt.xlabel('Time units')
    plt.ylabel('Top-5 Accuracy')

    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'acc_moco_{comm_time}.png')

