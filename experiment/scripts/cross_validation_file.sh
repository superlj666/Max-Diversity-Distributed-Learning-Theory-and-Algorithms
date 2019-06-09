if [ $# -lt 3 ]; then
    echo "usage: $0 file_name train_size splits"
    exit -1;
fi
file_name=$1
train_size=$2
bacth_size=$[$2/$3]
root_path=($(pwd)/data/test1)

# mkdir $root_path/meta/$file_name
# mv $root_path/meta/$file_name* $root_path/meta/$file_name

for element1 in $(ls $root_path/meta/$file_name);do
    mkdir $root_path/$element1

    split -l $train_size $root_path/meta/$file_name/$element1 -d -a 1 $root_path/meta/$file_name/$element1-

    mv $root_path/meta/$file_name/$element1-0 $root_path/$element1/$element1-0
    split -l $bacth_size $root_path/$element1/$element1-0 -d -a 4 $root_path/$element1/train_
    mv $root_path/$element1/$element1-0 $root_path/${element1}_train

    mv $root_path/meta/$file_name/$element1-1 $root_path/$element1/test_0000
done
