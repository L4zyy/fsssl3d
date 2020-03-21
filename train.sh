#!/bin/bash
if (($# != 2)); then
    echo "Usage: ./train.sh [0-1] [1-12], e.g. ./train.sh 1 1. First number is different configs and second number is view number."
    exit
fi

conf=$1
view=$2

episode=100
epoch=100

if [[ ${conf} == "0" ]]; then
    name="${view}view-1shot-5way"
    shot=1
    way=5
elif [[ ${conf} == "1" ]]; then
    name="${view}view-5shot-5way"
    shot=5
    way=5
else
    echo "Usage: ./train.sh [0-1] [1-12], e.g. ./train.sh 1 1. First number is different configs and second number is view number."
    exit
fi


python -m tools.train \
    --cuda \
    --name ${name} \
    --num_views ${view} \
    --num_query ${shot} \
    --num_support ${shot} \
    --num_way ${way} \
    --num_episode ${episode} \
    --num_epoches ${epoch} \
