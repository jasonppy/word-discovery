
model=$1 # disc-81
tgt_layer_for_attn=$2 # 7
k=$3 # 4096
threshold=$4 # 0.9
reduce_method=$5 # mean
segment_method=$6 # clsAttn or forceAlign
seed=$7
dataset=spokencoco # spokencoco

model_root=/data/scratch/pyp/exp_pyp
data_root=/data/scratch/pyp/datasets/coco_pyp
save_root=/data/scratch/pyp/exp_pyp

model_root=/data/scratch/pyp/exp_pyp
data_root=/data/scratch/pyp/datasets/coco_pyp
save_root=/data/scratch/pyp/exp_pyp

data_json="${data_root}/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
python ../save_seg_feats.py \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
--exp_dir ${model_root}/discovery/${model} \
--audio_base_path ${data_root}/SpokenCOCO \
--save_root ${save_root} \
--data_json ${data_json} \
--dataset ${dataset}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
python ../run_kmeans.py \
--seed ${seed} \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
-f "CLUS${k}" \
--exp_dir "${save_root}/${model}" \
--dataset ${dataset}


source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
python ../unit_analysis.py \
--exp_dir "${save_root}/discovery/word_unit_discovery/${model}/${dataset}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}" \
--data_json ${data_json} \
--k ${k} >> "./logs/${model}_${dataset}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}.log" 2>&1