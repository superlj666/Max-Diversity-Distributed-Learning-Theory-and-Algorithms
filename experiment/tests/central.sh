if [ $# -lt 4 ]; then
    echo "usage: $0 type file_name lambda feature_size"
    exit -1;
fi

for((i=1;i<=3;i++));do
file_name=$2"${i}"
train_path=($(pwd)/data/$1/${file_name}_train)
test_path=($(pwd)/data/$1/${file_name}/test_00)
train_size=$(cat ${train_path} | wc -l)
test_size=$(cat ${test_path} | wc -l)
feature_size=$4

if [ ! -d log/${file_name} ];then
mkdir log/${file_name}
fi

start_tm=`date +%s%N`
tests/test_rr_central ${file_name} ${train_path} ${test_path} ${train_size} ${test_size} ${feature_size} $3 > log/${file_name}/rr_central.log
echo $(tail -1 log/${file_name}/rr_central.log)
end_tm=`date +%s%N`
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}'`
echo "RR central cost time:" $use_tm



start_tm=`date +%s%N`
tests/test_krr_central ${file_name} ${train_path} ${test_path} ${train_size} ${test_size} ${feature_size} $3 > log/${file_name}/krr_central.log
echo $(tail -1 log/${file_name}/krr_central.log)

end_tm=`date +%s%N`
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}'`
echo "KRR central cost time:" $use_tm
echo
done

