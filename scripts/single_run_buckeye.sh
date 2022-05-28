
model=$1 # vg-hubert_3 or vg-hubert_4
tgt_layer_for_attn=$2 # 9
threshold=$3 # 0.7
reduce_method=$4 # mean, this won't affect results
segment_method=$5 # clsAttn
machine=$6
dataset=${7} # (lowercased) buckeyeval or buckeyetest

exp_dir_prefix=/data/scratch/pyp/exp_pyp
save_root_prefix=/data/scratch/pyp/exp_pyp
data_root=/data1/scratch/datasets_pyp/Buckeye/vad


source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
python ../save_seg_feats_buckeye.py \
--data_root ${data_root} \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
--exp_dir ${exp_dir_prefix}/discovery/${model} \
--save_root "${save_root_prefix}/discovery/word_unit_discovery/" \
--dataset ${dataset}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
python ../herman_eval.py \
--data_root ${data_root} \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
--dataset ${dataset}

