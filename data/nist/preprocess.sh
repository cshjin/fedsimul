NAME="nist" # name of the dataset, equivalent to directory name

cd ../../utils

# ./preprocess.sh -s niid --sf 0.05 -k 64 -t sample
# ./preprocess.sh --name nist -s niid --sf 1.0 -k 0 -t sample
./preprocess.sh --name $NAME $@

cd ../data/$NAME