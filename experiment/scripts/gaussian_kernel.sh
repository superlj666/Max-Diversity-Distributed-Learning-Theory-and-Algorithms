if [ $# -lt 2 ]; then
    echo "usage: $0 file_name sigma"
    exit -1;
fi

start_tm=`date +%s%N`;

file_name=${1}
file_path=$(pwd)/data/test1/${file_name}
sigma=$2

if [ ! -d $(pwd)/data/test1/kernel/${file_name} ];then
mkdir $(pwd)/data/test1/kernel/${file_name}
fi

for element1 in $(ls ${file_path});do
    for element2 in $(ls ${file_path});do
        if [ $element1 == "test_00" ];then
            continue
        fi
        left_file=${file_path}/${element1} 
        right_file=${file_path}/${element2} 
        left_size=$(cat ${file_path}/${element1}  | wc -l) 
        right_size=$(cat ${file_path}/${element2} | wc -l) 
        feature_size=$[$(head -1 ${right_file} | awk  '{print NF}')-1]
        save_path=$(pwd)/data/test1/kernel/${file_name}/${element1}-${element2}
        #echo $left_file $right_file $left_size $right_size $feature_size $save_path
        $(pwd)/tests/test_gaussian_kernel $left_file $right_file $left_size $right_size $feature_size $sigma $save_path &
    done
done

wait
end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}'`
echo "Cost time:" $use_tm