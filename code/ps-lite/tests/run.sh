#!/bin/bash
# set -x
if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi

export FEATURE_SIZE=90
export MAX_ITERATION=100
export ZETA=0.000001
export LAMBDA=0.02
export GAMMA=0.01
export FILE="YearPredictionMSD"

role=$1
shift
export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift
rank=$1
shift
file_name=($(pwd)/$1)
export SAMPLE_SIZE=$(cat $file_name | wc -l)
shift
bin=$1
shift
arg="$@"
export DMLC_LOCAL=1
# start the scheduler
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8000

if [ "$role" == "scheduler" ];then 
export DMLC_ROLE='scheduler'
echo "run scheduler"
${bin} ${arg} &

elif [ "$role" == "server" ];then
export DMLC_ROLE='server'
export HEAPPROFILE=./S$rank
${bin} ${arg} &
echo "run server"

else
export DMLC_ROLE='worker'
export HEAPPROFILE=./W$rank
${bin} ${arg} $file_name&
echo "run worker"
fi

wait
