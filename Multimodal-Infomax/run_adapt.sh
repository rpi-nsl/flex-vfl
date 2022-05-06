#!/bin/bash 

seed=$1
alg=$2
server_time=$3

python main_adapt.py --dataset mosei --contrast --alg $alg --seed $seed --num_epochs 100 --server_time $server_time --momentum 0.0
