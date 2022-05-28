model=$1 # 25
feats_type=$2 # preFeats or curFeats
k=$3 # 4096
threshold=$4 # 0.9
start_layer=$5 # 0
end_layer=$6 # 7 or 11 (for disc-37)
reduce_method=$7
segment_method=$8
num_runs=10
machine=$9
feature_extraction=${10} # 0 or 1, 0 means have already extacted the features in previous exp before, don't have to do it again
dataset=${11} # (lowercase) spokecoco, timit


if [[ ${start_layer} -eq 0 ]]; then
    bash sweep_single_run.sh $model curFeats 0 $k $threshold ${reduce_method} ${segment_method} ${num_runs} ${machine} ${feature_extraction} $dataset
    let start_layer=1
fi

for layer in $(seq ${start_layer} 1 ${end_layer}); do
    bash sweep_single_run.sh $model $feats_type $layer $k $threshold ${reduce_method} ${segment_method} ${num_runs} ${machine} ${feature_extraction} $dataset
done