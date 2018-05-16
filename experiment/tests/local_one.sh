#!/bin/bash
# set -x
if [ $# -lt 8 ]; then
    echo "usage: $0 lambda gamma num_servers num_workers file_name feature_size rr/krr mean/mdd "
    exit -1;
fi

export MAX_ITERATION=20
export ZETA=0.000001
export LAMBDA=$1
shift
export GAMMA=$1
shift
export SIGMA=2

export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift

file_name=$1
export DATA_PATH=$(pwd)/data/test1/$1/
export KERNEL_PATH=$(pwd)/data/test1/kernel/$1/
shift
export FEATURE_SIZE=$1
export TRAIN_SAMPLE_SIZE=$(cat ${DATA_PATH}train_00 | wc -l)
export TEST_SAMPLE_SIZE=$(cat ${DATA_PATH}test_00 | wc -l)

shift

kind=$1
if [ "$kind" == "rr" ];then 
bin="tests/test_rr_dist"
else
bin="tests/test_krr_dist"
fi
shift

method=$1
if [ "$method" == "mean" ];then 
export ZETA=1000000000000
fi

export DMLC_LOCAL=1
# start the scheduler
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8000
export DMLC_ROLE='scheduler'
${bin} &

if [ ! -d log/${file_name} ];then
mkdir log/${file_name}
fi

# start servers
export DMLC_ROLE='server'
for ((i=0; i<${DMLC_NUM_SERVER}; ++i)); do
    export HEAPPROFILE=./S${i}
    ${bin} > log/${file_name}/${kind}_${method}_server${i}.log & 
done

# start workers
export DMLC_ROLE='worker'
for ((i=0; i<${DMLC_NUM_WORKER}; ++i)); do
    export HEAPPROFILE=./W${i}
    ${bin} $i > log/${file_name}/${kind}_${method}_worker${i}.log &
done

# when finish starting nodes, starting count time
start_tm=`date +%s%N`;

wait

end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}'`
echo $(tail -1 log/${file_name}/${kind}_${method}_server0.log)
echo "Cost time:" $use_tm
# scheduler tests/run.sh scheduler 1 2 0 abalone tests/test_dist_rr_dc
# server tests/run.sh server 1 2 0 abalone tests/test_dist_rr_dc
# worker0 tests/run.sh worker 1 2 0 abalone tests/test_dist_rr_dc
# worker1 tests/run.sh worker 1 2 1 abalone tests/test_dist_rr_dc
