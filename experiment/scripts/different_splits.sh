if [ $# -lt 6 ]; then
    echo "usage: $0 file_name train_size feature_size lambda_rr lambda_krr gamma_krr"
    exit -1;
fi

file_name=$1
train_size=$2
feature_size=$3
lambda_rr=$4
lambda_krr=$5
gamma_krr=$6


splits=(10)
for split in ${splits[@]};do
    rm -rf $(pwd)/data/test1/$file_name*
    rm -rf $(pwd)/data/test1/kernel/$file_name*
    sh $(pwd)/scripts/cross_validation_file.sh $file_name $train_size $split
    sh $(pwd)/scripts/gaussian_kernel.sh ${file_name}1 2
    sh tests/central.sh $file_name 1 $feature_size $lambda_rr 1 $split $lambda_krr $gamma_krr
done