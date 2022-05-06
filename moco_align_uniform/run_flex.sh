#!/bin/bash 

seed=$1
mode=$2
st=$3
lr=$4
port=$5

if [[ "$2" == "flex" ]] || [[ "$2" == "sync" ]] || [[ "$2" == "sync1" ]]; then
    python main_flex.py -a resnet50 --lr $lr --batch-size 256 --schedule 30 60 90 --mode $mode --server_time $st --epochs 1000 --seed $seed --world-size 1 --rank 0 -j 16 --multiprocessing-distributed imagenet100/ --dist-url tcp://localhost:$port
else
    python main_flex.py -a resnet50 --lr $lr --batch-size 256 --schedule 150 300 500 --mode $mode --server_time $st --epochs 1000 --seed $seed --world-size 1 --rank 0 -j 16 --multiprocessing-distributed imagenet100/ --dist-url tcp://localhost:$port
fi
