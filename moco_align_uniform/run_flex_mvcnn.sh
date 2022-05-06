#!/bin/bash 

seed=$1
mode=$2
st=$3
port=$4

python main_flex_MVCNN.py -a resnet50 --lr 0.0001 --batch-size 64 --mode $mode --server_time $st --epochs 2000 --seed $seed --world-size 1 --rank 0 -j 16 --multiprocessing-distributed view/classes/ --dist-url tcp://localhost:$port 
