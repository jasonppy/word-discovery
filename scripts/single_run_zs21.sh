
model=$1 #vg-hubert_3 or vg-hubert_4
tgt_layer_for_attn=$2 # 9
k=$3 # 4096
threshold=$4 # 0.7
reduce_method=$5 # weightedmean
segment_method=$6 # clsAttn
seed=$7
dataset=zs21

exp_dir_prefix=/data1/scratch/exp_pyp
save_root_prefix=/data1/scratch/exp_pyp
data_root=/data2/scratch/pyp/datasets/vads
audio_base_path=/data2/scratch/pyp/datasets/2020/2017/english/train


source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
python ../save_seg_feats_zs21.py \
--data_root ${data_root} \
--audio_base_path ${audio_base_path} \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
--exp_dir ${exp_dir_prefix}/discovery/${model} \
--save_root "${save_root_prefix}/discovery/word_unit_discovery/" \
--dataset ${dataset}


source ~/miniconda3/etc/profile.d/conda.sh
conda activate faiss_env
python ../run_kmeans.py \
--seed ${seed} \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
-f "CLUS${k}" \
--exp_dir "${save_root_prefix}/discovery/word_unit_discovery/${model}" \
--dataset ${dataset}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
python ../prepare_zs21_submit.py \
--exp_dir "${save_root_prefix}/discovery/word_unit_discovery/${model}/${dataset}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}" \
--k ${k}

echo "run zerospeech 2020 evaluation"
echo "run zerospeech 2020 evaluation"
echo "run zerospeech 2020 evaluation"
echo "run zerospeech 2020 evaluation"
cd ~/zerospeech2020
bash eval_zs2017_track2.sh "${model}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}_${k}"