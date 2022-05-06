import numpy as np
import os
import random

seeds = [707412115,1928644128,16910772,1263880818,1445547577]
server_times = ["1.0","10.0","50.0"]
port = 10001
# ImageNet
for seed in seeds:
    os.system(f'./run_flex.sh {seed} flex 1.0 0.03 {port}')
    os.system(f'./run_flex.sh {seed} sync 1.0 0.03 {port+1}')
    os.system(f'./run_flex.sh {seed} sync1 1.0 0.03 {port+2}')
    os.system(f'./run_flex.sh {seed} pbcd 1.0 0.03 {port+3}')
    port += 4
    for server_time in server_times:
        os.system(f'./run_flex.sh {seed} vafl {server_time} 0.03 {port}')
        port += 1

# ModelNet
for seed in seeds:
    os.system(f'./run_flex_mvcnn.sh {seed} flex2 1.0 {port+1}')
    os.system(f'./run_flex_mvcnn.sh {seed} flex3 1.0 {port+2}')
    os.system(f'./run_flex_mvcnn.sh {seed} sync 1.0 {port+3}')
    os.system(f'./run_flex_mvcnn.sh {seed} sync1 1.0 {port+4}')
    os.system(f'./run_flex_mvcnn.sh {seed} sync2 1.0 {port+5}')
    os.system(f'./run_flex_mvcnn.sh {seed} sync3 1.0 {port+6}')
    os.system(f'./run_flex_mvcnn.sh {seed} pbcd 1.0 {port+7}')
    port += 8
    for server_time in server_times:
        os.system(f'./run_flex_mvcnn.sh {seed} vafl {server_time} {port}')
        os.system(f'./run_flex_mvcnn.sh {seed} vafl2 {server_time} {port+1}')
        os.system(f'./run_flex_mvcnn.sh {seed} vafl3 {server_time} {port+2}')
        port += 3
    os.system(f'./run_flex_mvcnnadapt.sh {seed} adapt {port}')
    port += 1
    os.system(f'./run_flex_mvcnnadapt.sh {seed} nonadapt {port}')
    port += 1
