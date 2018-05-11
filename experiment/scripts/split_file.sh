if [ $# -lt 2 ]; then
    echo "usage: $0 target_name all_file"
    echo "or"
    echo "usage: $0 target_name train_file test_file"
    exit -1;
fi

file_path=($(pwd)/data/meta/$2)
echo ${file_path}

sample_size=$(cat ${file_path}|wc -l)
echo ${sample_size}

# if more than 5*20000
if [ $# -lt 3 ]; then
    if [ ${sample_size} -gt 100000 ];then 
        splits=$[${sample_size}/20000]
        test_size=$[${sample_size}%20000]
        split_size=20000
    else
        splits=5
        echo ${splits}
        test_size=$[${sample_size}/3]
        split_size=$[(${sample_size}-test_size)/5]
    fi

    echo ${splits}
    echo ${test_size}
    echo ${split_size}
    
    split -l $[${sample_size} - ${test_size}] ${file_path} -d -a 2 ${file_path}_
    divided_path=($(pwd)/data/$1)
    mkdir ${divided_path}
    mv ${file_path}_00 ${divided_path}/train_all
    mv ${file_path}_01 ${divided_path}/test
    split -l ${split_size} ${divided_path}/train_all -d -a 2 ${divided_path}/train_
else
    if [ ${sample_size} -gt 100000 ];then 
        split_size=20000
    else
        splits=5
        split_size=$[${sample_size}/5]
    fi
    
    divided_path=($(pwd)/data/$1)
    mkdir ${divided_path}
    test_path=($(pwd)/data/meta/$3)
    echo ${test_path}

    cp ${file_path} ${divided_path}/train_all
    cp ${test_path} ${divided_path}/test
    split -l ${split_size} ${divided_path}/train_all -d -a 2 ${divided_path}/train_
fi
