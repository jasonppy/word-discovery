import argparse
import torch
import os
import pickle
import json
import soundfile as sf
import tqdm
import time
from models import audio_encoder
import tqdm
from itertools import groupby
from operator import itemgetter
import csv
from collections import defaultdict
def cls_attn_seg_feats(feats, cls_attn_weights, threshold, pool, spf, level2, start_sec, end_sec):
    # return a list of features that are segmented by cls attn weights
    threshold_value = torch.quantile(cls_attn_weights, threshold, dim=-1, keepdim=True) # [n_h, T]
    cls_attn_weights_sum = cls_attn_weights.sum(0)
    important_idx = torch.where((cls_attn_weights >= threshold_value).float().sum(0) > 0)[0].cpu().numpy()
    boundaries = []
    boundaries_all = []
    boundaries_ex1 = []
    for k, g in groupby(enumerate(important_idx), lambda ix : ix[0] - ix[1]):
        seg = list(map(itemgetter(1), g))
        t_s, t_e = seg[0], min(seg[-1]+1, cls_attn_weights.shape[-1])
        if len(seg) > 1:
            boundaries_all.append([t_s, t_e])
            boundaries_ex1.append([t_s, t_e])
        else:
            boundaries_all.append([t_s, t_e])
    
    if level2 or len(boundaries_ex1) == 0:
        boundaries = boundaries_all
    else:
        boundaries = boundaries_ex1

    seg_feats = []
    locations = []
    boundaries_in_sec = []
    total_b = len(boundaries)
    # print(boundaries)
    for i, (t_s, t_e) in enumerate(boundaries):
        locations.append(start_sec + spf*(t_s+t_e)/2.) # in seconds
        if i == 0:
            boundaries_in_sec.append([start_sec, start_sec + t_e*spf]) # in seconds
        elif i == total_b - 1:
            boundaries_in_sec.append([start_sec + t_s*spf, end_sec]) # in seconds
        else:
            boundaries_in_sec.append([start_sec + t_s*spf, start_sec + t_e*spf]) # in seconds
        if pool == "mean":
            seg_feats.append(feats[t_s:t_e].mean(0).cpu())
        elif pool == "max":
            # print(t_s, t_e)
            max_id = torch.argmax(cls_attn_weights_sum[t_s:t_e])
            seg_feats.append(feats[t_s+max_id].cpu())
        elif pool == "median":
            seg_feats.append(feats[int((t_s+t_e)/2)].cpu())
        elif pool == "weightedmean":
            seg_feats.append((feats[t_s:t_e]*(cls_attn_weights_sum[t_s:t_e]/cls_attn_weights_sum[t_s:t_e].sum()).unsqueeze(1)).sum(0).cpu())
    return {"seg_feats": seg_feats, "locations": locations, "boundaries": boundaries_in_sec}


def force_align_seg_feats(feats, text_alignment, fps, pool):
    seg_feats = []
    locations = []
    boundaries = []
    meta_toks = text_alignment.split(" ")
    for meta_tok in meta_toks:
        toks = meta_tok.split('__')
        if len(toks) == 3:
            s = float(toks[0])
            e = float(toks[2])
            boundaries.append([s,e])
            locations.append((s+e)/2.)
            if pool == "mean":
                seg_feats.append(feats[int(s*fps):int(e*fps)].mean(0).cpu())
            elif pool == "max":
                seg_feats.append(feats[int(s*fps):int(e*fps)].max(0)[0].cpu())
            elif pool == "median":
                seg_feats.append(feats[int((s*fps+e*fps)/2)].cpu())
    return {"seg_feats": seg_feats, "locations": locations, "boundaries": boundaries}
    


print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_dir", type=str)
parser.add_argument("--dataset", type=str, default='zs21')
parser.add_argument("--data_root", type=str, default="/data2/scratch/pyp/datasets/vads")
parser.add_argument("--audio_base_path", type=str, default="/data2/scratch/pyp/datasets/2020/2017/english/train")
parser.add_argument("--save_root", type=str, default="/data2/scratch/pyp/discovery/word_unit_discovery/")
parser.add_argument("--percentage", type=int, default=None, help="if None, the feats_type is the original name, otherwise, it's feats_type_percentage")
parser.add_argument("--threshold", type=float, default=0.90)
parser.add_argument("--reduce_method", type=str, default="mean", choices=['mean', 'max', 'median', 'weightedmean'])
parser.add_argument("--tgt_layer_for_attn", type=int, default=7, help="where attn weights are coming from, as for features, if feats_type==preFeats, and feature comes from previous layer of tgt_layer_for_attn, otherwise, feature comes from the same layer")
parser.add_argument("--level2", action="store_true", default=False, help="if True, use feats and atten weights from level2 (not avaliable for models that only has one level of w2v2)")
parser.add_argument("--segment_method", type=str, choices=['clsAttn', 'forceAlign'], default=None, help="if use cls attn segmentation or use force alignment segmentation. If use, need model_args.use_audio_cls_token to be True")
args = parser.parse_args()
feats_type = args.dataset + "_" + args.reduce_method + "_" + str(args.threshold) + "_" + str(args.tgt_layer_for_attn) + "_" + args.segment_method
save_root = os.path.join(args.data_root, args.exp_dir.split("/")[-1], feats_type)
print("data save at: ", save_root)
os.makedirs(save_root, exist_ok=True)
print(args)
if not os.path.isdir(args.exp_dir):
    raise RuntimeError(f"{args.exp_dir} does not exist!!")

########################## setup model ##########################
with open(os.path.join(args.exp_dir, "args.pkl"), "rb") as f:
    model_args = pickle.load(f)
model = audio_encoder.AudioEncoder(model_args)
bundle = torch.load(os.path.join(args.exp_dir, "best_bundle.pth"))
model.carefully_load_state_dict(bundle['dual_encoder'], load_all=True)
model.eval()
model = model.cuda()
########################## setup model ##########################


data_start_time = time.time()


locF_temp = []
j = 0
# total_data = []
data_dict = {}
missing_ali = 0
level2 = False
tgt_layer = args.tgt_layer_for_attn
all_data = defaultdict(list)
if not os.path.isfile(os.path.join(args.data_root,"2020/ENGLISH_VAD.pkl")):
    with open(os.path.join(args.data_root,"2020/ENGLISH_VAD.csv"), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for i, line in enumerate(reader):
            line = line[0].split(",")
            # if line
            all_data[line[0]].append([float(line[1]), float(line[2])])

    with open(os.path.join(args.data_root,"2020/ENGLISH_VAD.pkl"), "wb") as f:
        pickle.dump(all_data, f)
else:
    with open(os.path.join(args.data_root,"2020/ENGLISH_VAD.pkl"), "rb") as f:
        all_data = pickle.load(f)

audio_base_path = os.path.join(args.data_root, "2020/2017/english/train")
for key in tqdm.tqdm(all_data.keys()):
    pointer = 0
    wav_fn = os.path.join(audio_base_path, key+".wav")
    if not os.path.isfile(wav_fn):
        print(f"{wav_fn} not found")
        continue
    total_audio, sr = sf.read(wav_fn, dtype = 'float32')
    total_audio = torch.from_numpy(total_audio).unsqueeze(0).cuda()
    assert sr == 16000
    # for start_sec, end_sec in all_data[key]:
    while pointer < len(all_data[key]):
        start_sec, end_sec = all_data[key][pointer]
        while end_sec - start_sec < 2:
            pointer += 1
            if pointer < len(all_data[key]):
                new_start_sec, end_sec = all_data[key][pointer]
            else:
                break
        if end_sec - start_sec < .5:
            break
        pointer += 1
        audio_use = total_audio[:, int(start_sec*sr):int(end_sec*sr+1)]
        with torch.no_grad():
            w2v2_out = model(audio_use, padding_mask=None, mask=False, need_attention_weights=True, tgt_layer=tgt_layer)
        
        if args.segment_method == "clsAttn": # use cls attn for segmentation
            assert model_args.use_audio_cls_token and model_args.cls_coarse_matching_weight > 0.
            feats = w2v2_out['features'].squeeze(0)[1:] # [1, T+1, D] -> [T, D]
            spf = audio_use.shape[-1]/sr/feats.shape[-2]
            attn_weights = w2v2_out['attn_weights'].squeeze(0) # [1, num_heads, tgt_len, src_len] -> [num_heads, tgt_len, src_len]
            cls_attn_weights = attn_weights[:, 0, 1:] # [num_heads, tgt_len, src_len] -> [n_h, T]
            out = cls_attn_seg_feats(feats, cls_attn_weights, args.threshold, args.reduce_method, spf, level2, start_sec, end_sec)
        else:
            raise NotImplementedError
        
        seg_feats = out['seg_feats']
        seg_feats = torch.stack(seg_feats).cpu()
        
        data_dict[f"{key}_{start_sec:.2f}-{end_sec:.2f}"] = {"seg_feats": seg_feats, "locations": torch.tensor(out['locations']), "boundaries": torch.tensor(out['boundaries']), "spf":spf}

if args.segment_method == "forceAlign":
    print(f"missing alignments: {missing_ali}")

with open(os.path.join(save_root, 'data_dict.pkl'), "wb") as f:
    pickle.dump(data_dict, f)
print(f"save pickle data at {os.path.join(save_root, 'data_dict.pkl')}")


