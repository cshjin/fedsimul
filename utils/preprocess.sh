#!/usr/bin/env bash
#
# script to preprocess data

# TODO: change the bash script to python file

# --------------------
# parse arguments

NAME="nist"  # --name tag, name of the dataset, equivalent to directory name
SAMPLE="na"     # -s tag, iid or niid
IUSER=""        # --iu tag, # of users if iid sampling
SFRAC=""        # --sf tag, fraction of data to sample
MINSAMPLES="na" # -k tag, minimum allowable # of samples per user
TRAIN="na"      # -t tag, user or sample
TFRAC=""        # --tf tag, fraction of data in training set

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --name)
    NAME="$2"
    shift # past argument
    if [ ${SAMPLE:0:1} = "-" ]; then
        NAME="nist"
    else
        shift # past value
    fi
    ;;
    -s)
    SAMPLE="$2"
    shift # past argument
    if [ ${SAMPLE:0:1} = "-" ]; then
        SAMPLE=""
    else
        shift # past value
    fi
    ;;
    --iu)
    IUSER="$2"
    shift # past argument
    if [ ${IUSER:0:1} = "-" ]; then
        IUSER=""
    else
        shift # past value
    fi
    ;;
    --sf)
    SFRAC="$2"
    shift # past argument
    if [ ${SFRAC:0:1} = "-" ]; then
        SFRAC=""
    else
        shift # past value
    fi
    ;;
    -k)
    MINSAMPLES="$2"
    shift # past argument
    if [ ${MINSAMPLES:0:1} = "-" ]; then
        MINSAMPLES=""
    else
        shift # past value
    fi
    ;;
    -t)
    TRAIN="$2"
    shift # past argument
    if [ -z "$TRAIN" ] || [ ${TRAIN:0:1} = "-" ]; then
        TRAIN=""
    else
        shift # past value
    fi
    ;;
    --tf)
    TFRAC="$2"
    shift # past argument
    if [ ${TFRAC:0:1} = "-" ]; then
        TFRAC=""
    else
        shift # past value
    fi
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

# --------------------
# preprocess data

CONT_SCRIPT=true
cd /tmp/data/$NAME

# download data and convert to .json format

# if [ ! -d "data/all_data" ]; then
#     cd preprocess
#     ./data_to_json.sh
#     cd ..
# fi

NAMETAG="--name $NAME"

# sample data
IUSERTAG=""
if [ ! -z $IUSER ]; then
    IUSERTAG="--u $IUSER"
fi
SFRACTAG=""
if [ ! -z $SFRAC ]; then
    SFRACTAG="--fraction $SFRAC"
fi

if [ "$CONT_SCRIPT" = true ] && [ ! $SAMPLE = "na" ]; then
    if [ -d "sampled_data" ] && [ "$(ls -A sampled_data)" ]; then
        CONT_SCRIPT=false
    else
        if [ ! -d "sampled_data" ]; then
            mkdir sampled_data
        fi

        # cd ../../utils

        if [ $SAMPLE = "iid" ]; then
            python3 sample.py $NAMETAG --iid $IUSERTAG $SFRACTAG
        else
            python3 sample.py $NAMETAG $SFRACTAG
        fi

        cd /tmp/data/$NAME
    fi
fi

# remove users with less then given number of samples
if [ "$CONT_SCRIPT" = true ] && [ ! $MINSAMPLES = "na" ]; then
    if [ -d "rem_user_data" ] && [ "$(ls -A rem_user_data)" ]; then
        CONT_SCRIPT=false
    else
        if [ ! -d "rem_user_data" ]; then
            mkdir rem_user_data
        fi

        cd ../../utils

        if [ -z $MINSAMPLES ]; then
            python3 remove_users.py $NAMETAG
        else
            python3 remove_users.py $NAMETAG --min_samples $MINSAMPLES
        fi

        cd /tmp/data/$NAME
    fi
fi

# create train-test split
TFRACTAG=""
if [ ! -z $TFRAC ]; then
    TFRACTAG="--frac $TFRAC"
fi

if [ "$CONT_SCRIPT" = true ] && [ ! $TRAIN = "na" ]; then
    if [ -d "train" ] && [ "$(ls -A train)" ]; then
        CONT_SCRIPT=false
    else
        if [ ! -d "train" ]; then
            mkdir train
        fi
        if [ ! -d "test" ]; then
            mkdir test
        fi

        cd ../../utils

        if [ -z $TRAIN ]; then
            python3 split_data.py $NAMETAG $TFRACTAG
        elif [ $TRAIN = "user" ]; then
            python3 split_data.py $NAMETAG --by_user $TFRACTAG
        elif [ $TRAIN = "sample" ]; then
            python3 split_data.py $NAMETAG --by_sample $TFRACTAG
        fi

        cd ../data/$NAME
    fi
fi

if [ "$CONT_SCRIPT" = false ]; then
    echo "Data for one of the specified preprocessing tasks has already been"
    echo "generated. If you would like to re-generate data for this directory,"
    echo "please delete the existing one. Otherwise, please remove the"
    echo "respective tag(s) from the preprocessing command."
fi