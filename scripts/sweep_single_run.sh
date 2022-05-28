
model=$1 # 26
tgt_layer_for_attn=$2 # 7
k=$3 # 4096
threshold=$4 # 0.9
reduce_method=$5
segment_method=$6
num_runs=$7
machine=$8
feature_extraction=${9}
dataset=${10}


exp_dir_prefix=/data/scratch/pyp/exp_pyp
audio_base_path_prefix=/data/scratch/pyp/datasets/coco_pyp
save_root_prefix=/data/scratch/pyp/exp_pyp
data_json_prefix=/data/scratch/pyp/datasets/coco_pyp

data_json=${data_json_prefix}/SpokenCOCO/SpokenCOCO_test_unrolled_karpathy_with_alignments.json


if [[ ${feature_extraction} -eq 1 ]]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate tf2
    python ../save_seg_feats.py \
    --segment_method ${segment_method} \
    --threshold ${threshold} \
    --reduce_method ${reduce_method} \
    --tgt_layer_for_attn ${tgt_layer_for_attn} \
    --exp_dir ${exp_dir_prefix}/discovery/${model} \
    --audio_base_path ${audio_base_path_prefix}/SpokenCOCO \
    --save_root "${save_root_prefix}/discovery/test_word_unit_discovery/" \
    --data_json ${data_json} \
    --dataset ${dataset}
fi

for seed in $(seq 1 1 ${num_runs}); do
    ln -s "${save_root_prefix}/discovery/test_word_unit_discovery/${model}" "${save_root_prefix}/discovery/test_word_unit_discovery/${model}_seed${seed}"
    
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate tf2
    python ../run_kmeans.py \
    --seed ${seed} \
    --segment_method ${segment_method} \
    --threshold ${threshold} \
    --reduce_method ${reduce_method} \
    --tgt_layer_for_attn ${tgt_layer_for_attn} \
    -f "CLUS${k}" \
    --exp_dir "${save_root_prefix}/discovery/test_word_unit_discovery/${model}_seed${seed}" \
    --dataset ${dataset}

    echo "" >> "./final_logs/${model}_${dataset}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}.log" 2>&1
    echo "seed${seed}" >> "./final_logs/${model}_${dataset}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}.log" 2>&1
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate tf2
    python ../unit_analysis.py \
    --exp_dir "${save_root_prefix}/discovery/test_word_unit_discovery/${model}_seed${seed}/${dataset}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}" \
    --data_json ${data_json} \
    --k ${k} >> "./final_logs/${model}_${dataset}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}.log" 2>&1
done