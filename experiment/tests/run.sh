#!/bin/bash
# set -x
if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi
start_tm=`date +%s%N`;

export MAX_ITERATION=10
export ZETA=1000
export LAMBDA=0.02
export GAMMA=0.01

role=$1
shift
export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift
rank=$1
shift
test_path_prefix=($(pwd)/data/$1/test)
train_path_prefix=($(pwd)/data/$1/train)

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

${bin} ${arg} ${test_path_prefix} $(cat ${test_path_prefix} | wc -l) $[$(head -1 ${test_path_prefix} | awk  '{print NF}')-1]&
echo "run server"

else
export DMLC_ROLE='worker'
export HEAPPROFILE=./W$rank
${bin} ${arg} ${train_path_prefix}_$rank $(cat ${train_path_prefix}_$rank | wc -l) $[$(head -1 ${train_path_prefix}_$rank | awk  '{print NF}')-1]&
echo "run worker"
echo ${train_path_prefix}_$rank
echo $(cat ${train_path_prefix}_$rank | wc -l)
echo $(head -1 ${train_path_prefix}_$rank | awk  '{print NF}')
fi

wait

end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}'`
echo "Cost time:" $use_tm
# scheduler tests/run.sh scheduler 1 2 0 abalone tests/test_krr_dist
# server tests/run.sh server 1 2 0 abalone tests/test_krr_dist
# worker0 tests/run.sh worker 1 2 0 abalone tests/test_krr_dist
# worker1 tests/run.sh worker 1 2 1 abalone tests/test_krr_dist

#!/bin/bash
# set -x
if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi
start_tm=`date +%s%N`;

export MAX_ITERATION=50
export ZETA=0.000001
export LAMBDA=0.02
export GAMMA=0.01
export SIGMA=2

export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift

file_name=$1
test_path_prefix=($(pwd)/data/$1/)
train_path_prefix=($(pwd)/data/$1/train)
export FEATURE_SIZE=$[$(head -1 ${test_path_prefix}test | awk  '{print NF}')-1]
export TRAIN_SAMPLE_SIZE=$(cat ${train_path_prefix}_all | wc -l)
export TEST_SAMPLE_SIZE=$(cat ${test_path_prefix}test | wc -l)

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
export ZETA=10000
fi

export DMLC_LOCAL=1
# start the scheduler
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8000
export DMLC_ROLE='scheduler'
${bin} &

# start servers
export DMLC_ROLE='server'
for ((i=0; i<${DMLC_NUM_SERVER}; ++i)); do
    export HEAPPROFILE=./S${i}
    ${bin} ${test_path_prefix} > log/${file_name}/${kind}_${method}_server${i}.log&
done

# start workers
export DMLC_ROLE='worker'
for ((i=0; i<${DMLC_NUM_WORKER}; ++i)); do
    export HEAPPROFILE=./W${i}
    ${bin} ${train_path_prefix}_${i} $(cat ${train_path_prefix}_${i} | wc -l) $[$(head -1 ${train_path_prefix}_${i} | awk  '{print NF}')-1]  > log/${file_name}/${kind}_${method}_worker${i}.log&
done

wait

end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}'`
echo $(tail -1 log/${file_name}/${kind}_${method}_server0.log)
echo "Cost time:" $use_tm