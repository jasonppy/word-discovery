model=$1 # 25
feats_type=$2 # preFeats or curFeats
k=$3 # 4096
threshold=$4 # 0.9
start_layer=$5 # 0
end_layer=$6 # 7 or 11 (for disc-37)
reduce_method=$7
segment_method=$8
seed=4
machine=$9
dataset=${10} # (lowercase) spokecoco, timit

if [[ ${start_layer} -eq 0 ]]; then
    bash single_run.sh $model curFeats 0 $k $threshold ${reduce_method} ${segment_method} ${seed} ${machine} $dataset
    let start_layer=1
fi

for layer in $(seq ${start_layer} 1 ${end_layer}); do
    bash single_run.sh $model $feats_type $layer $k $threshold ${reduce_method} ${segment_method} ${seed} ${machine} $dataset
done