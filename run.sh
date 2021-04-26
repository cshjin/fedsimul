#!/bin/sh

ds=${1:-mnist}
mu=${2:-0}
python3  -u main.py \
        --dataset=$ds \
        --optimizer='fedmom'  \
        --learning_rate=0.01 \
        --num_rounds=200 \
        --clients_per_round=10 \
        --mu=$mu \
        --eval_every=1 \
        --batch_size=10 \
        --num_epochs=20 \
        --model='mclr'
