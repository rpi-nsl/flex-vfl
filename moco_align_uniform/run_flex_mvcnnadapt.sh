#!/bin/bash 

seed=$1
mode=$2
port=$3

python main_flex_MVCNNadapt.py -a resnet50 --lr 0.0001 --batch-size 64 --mode $mode --epochs 200 --seed $seed --world-size 1 --rank 0 -j 1 view/classes/ --dist-url tcp://localhost:$port 
