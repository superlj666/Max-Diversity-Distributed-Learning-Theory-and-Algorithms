if [ $# -lt 8 ]; then
    echo "usage: $0 file_name folds feature_size lambda_rr servers workers lambda_krr gamma"
    exit -1;
fi

folds=$2
feature_size=$3
lambda_rr=$4
servers=$5
workers=$6
lambda_krr=$7
gamma=$8

log_file_rr_central=log/$1_rr_central.log
rm -rf $log_file_rr_central
log_file_krr_central=log/$1_krr_central.log
rm -rf $log_file_krr_central
log_file_rr_mean=log/$1_rr_mean.log
rm -rf $log_file_rr_mean
log_file_rr_mdd=log/$1_rr_mdd.log
rm -rf $log_file_rr_mdd
log_file_krr_mean=log/$1_krr_mean.log
rm -rf $log_file_krr_mean
log_file_krr_mdd=log/$1_krr_mdd.log
rm -rf $log_file_krr_mdd

for((i=1;i<=$folds;i++));do
    file_name=$1"${i}"
    train_path=($(pwd)/data/test1/${file_name}_train)
    test_path=($(pwd)/data/test1/${file_name}/test_00)
    train_size=$(cat ${train_path} | wc -l)
    test_size=$(cat ${test_path} | wc -l)

    if [ ! -d log/${file_name} ];then
    mkdir log/${file_name}
    fi
    
    # rr central
    start_tm=`date +%s%N`
    tests/test_rr_central ${file_name} ${train_path} ${test_path} ${train_size} ${test_size} ${feature_size} $lambda_rr > log/${file_name}/rr_central.log
    MSE=$(tail -1 log/${file_name}/rr_central.log) 
    end_tm=`date +%s%N`
    use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}'`
    echo -n $MSE >> $log_file_rr_central
    echo -n " " >> $log_file_rr_central
    printf "%.6f" $use_tm >> $log_file_rr_central
    echo >> $log_file_rr_central

    # krr central
    start_tm=`date +%s%N`
    tests/test_krr_central ${file_name} ${train_path} ${test_path} ${train_size} ${test_size} ${feature_size} $lambda_rr > log/${file_name}/krr_central.log
    MSE=$(tail -1 log/${file_name}/krr_central.log)
    end_tm=`date +%s%N`
    use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}'`
    echo  -n $MSE >> $log_file_krr_central
    echo -n " " >> $log_file_krr_central
    printf "%.6f" $use_tm >> $log_file_krr_central
    echo >> $log_file_krr_central

    # rr mean
    sh tests/local_one.sh $lambda_krr $gamma $servers $workers ${file_name} $feature_size rr mean >> $log_file_rr_mean

    # rr mdd
    sh tests/local_one.sh $lambda_krr $gamma $servers $workers ${file_name} $feature_size rr mdd >> $log_file_rr_mdd

    # krr mean
    sh tests/local_one.sh $lambda_krr $gamma $servers $workers ${file_name} $feature_size krr mean >> $log_file_krr_mean

    # krr mdd
    sh tests/local_one.sh $lambda_krr $gamma $servers $workers ${file_name} $feature_size krr mdd >> $log_file_krr_mdd
    

done

