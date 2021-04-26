#!/usr/bin/env bash

NAME="shakespeare"

cd ../../utils

./preprocess.sh --name $NAME $@

cd ../data/$NAME