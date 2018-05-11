file_name=$1
train_path=($(pwd)/data/$1/train_all)
test_path=($(pwd)/data/$1/test)
train_size=$(cat ${train_path} | wc -l)
test_size=$(cat ${test_path} | wc -l)
feature_size=$[$(head -1 ${test_path} | awk  '{print NF}')-1]

start_tm=`date +%s%N`
tests/test_rr_central ${file_name} ${train_path} ${test_path} ${train_size} ${test_size} ${feature_size} 
end_tm=`date +%s%N`
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}'`
echo "RR central cost time:" $use_tm

echo

start_tm=`date +%s%N`
tests/test_krr_central ${file_name} ${train_path} ${test_path} ${train_size} ${test_size} ${feature_size} 
end_tm=`date +%s%N`
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}'`
echo "KRR central cost time:" $use_tm
