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
import numpy as np
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
def cls_attn_seg_feats(feats, cls_attn_weights, threshold, spf, level2, start_sec, end_sec):
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
    return {"locations": locations, "boundaries": boundaries_in_sec}


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
parser.add_argument("--dataset", type=str, default='buckeyeval', choices=['buckeyeval', 'buckeyetest'])
parser.add_argument("--data_root", type=str, default="/data1/scratch/datasets_pyp/Buckeye")
parser.add_argument("--save_root", type=str, default="/data2/scratch/pyp/discovery/word_unit_discovery/")
parser.add_argument("--percentage", type=int, default=None, help="if None, the feats_type is the original name, otherwise, it's feats_type_percentage")
parser.add_argument("--threshold", type=float, default=0.90)
parser.add_argument("--reduce_method", type=str, default="mean", choices=['mean', 'max', 'median', 'weightedmean'])
parser.add_argument("--tgt_layer_for_attn", type=int, default=7, help="where attn weights are coming from, as for features, if feats_type==preFeats, and feature comes from previous layer of tgt_layer_for_attn, otherwise, feature comes from the same layer")
parser.add_argument("--segment_method", type=str, choices=['clsAttn', 'forceAlign'], default=None, help="if use cls attn segmentation or use force alignment segmentation. If use, need model_args.use_audio_cls_token to be True")
args = parser.parse_args()

save_root = os.path.join(args.save_root, args.exp_dir.split("/")[-1])
feats_type = args.dataset + "_" + args.reduce_method + "_" + str(args.threshold) + "_" + str(args.tgt_layer_for_attn) + "_" + args.segment_method
if args.percentage is not None:
    feats_type = feats_type + "_" + str(args.percentage)
out_dir = os.path.join(args.data_root, feats_type)
print("data save at: ", out_dir)
os.makedirs(out_dir, exist_ok=True)
print(args)
if not os.path.isdir(args.exp_dir):
    raise RuntimeError(f"{args.exp_dir} does not exist!!")

########################## setup model ##########################
with open(os.path.join(args.exp_dir, "args.pkl"), "rb") as f:
    model_args = pickle.load(f)
model = audio_encoder.AudioEncoder(model_args)
bundle = torch.load(os.path.join(args.exp_dir, "best_bundle.pth"))
if "dual_encoder" in bundle:
    model.carefully_load_state_dict(bundle['dual_encoder'], load_all=True)
else:
    model.carefully_load_state_dict(bundle['model'], load_all=True)
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
split = "test" if "test" in args.dataset else "val"
if not os.path.isfile(os.path.join(args.data_root, "vad", f"{split}.pkl")):
    with open(os.path.join(args.data_root, "vad", f"{split}.json"), "r") as f:
        in_files = json.load(f)
        for i, item in enumerate(in_files):
            fn = item[0] # "s17/s1701a"
            s, e = float(item[1]), float(item[1]) + float(item[2]) # this is in seconds
            match_key = item[3].split("/")[-1]
            all_data[fn].append([[s,e], match_key]) # s01_01a_003222-003256

    with open(os.path.join(args.data_root, "vad", f"{split}.pkl"), "wb") as f:
        pickle.dump(all_data, f)
else:
    with open(os.path.join(args.data_root, "vad", f"{split}.pkl"), "rb") as f:
        all_data = pickle.load(f)


for key in tqdm.tqdm(all_data.keys()):
    pointer = 0
    wav_fn = os.path.join(args.data_root, key+".wav")
    if not os.path.isfile(wav_fn):
        print(f"{wav_fn} not found")
        continue
    total_audio, sr = sf.read(wav_fn, dtype = 'float32')
    total_audio = torch.from_numpy(total_audio).unsqueeze(0).cuda()
    assert sr == 16000
    # for start_sec, end_sec in all_data[key]:
    file_len = len(all_data[key])
    while pointer < file_len:
        [start_sec, end_sec], match_key = all_data[key][pointer]
        pointer += 1
        cur_match_keys = [match_key]
        while end_sec - start_sec < .05:
            if pointer < len(all_data[key]):
                [new_start_sec, end_sec], match_key = all_data[key][pointer]
                pointer += 1
                cur_match_keys.append(match_key)
            else:
                break
        if end_sec - start_sec < .05:
            # break
            [start_sec, new_end_sec], match_key = all_data[key][pointer-2]
            cur_match_keys = [match_key] + cur_match_keys
        audio_use = total_audio[:, int(start_sec*sr):int(end_sec*sr+1)]
        # print(start_sec, end_sec)
        # print(cur_match_keys)
        with torch.no_grad():
            w2v2_out = model(audio_use, padding_mask=None, mask=False, need_attention_weights=True, tgt_layer=tgt_layer)
        
        if args.segment_method == "clsAttn": # use cls attn for segmentation
            assert model_args.use_audio_cls_token and model_args.cls_coarse_matching_weight > 0.
            feats = w2v2_out['features'].squeeze(0)[1:] # [1, T+1, D] -> [T, D]
            spf = audio_use.shape[-1]/sr/feats.shape[-2]
            attn_weights = w2v2_out['attn_weights'].squeeze(0) # [1, num_heads, tgt_len, src_len] -> [num_heads, tgt_len, src_len]
            cls_attn_weights = attn_weights[:, 0, 1:] # [num_heads, tgt_len, src_len] -> [n_h, T]
            out = cls_attn_seg_feats(feats, cls_attn_weights, args.threshold, spf, level2, start_sec, end_sec)
        else:
            raise NotImplementedError
        
        # seg_feats = out['seg_feats']
        # seg_feats = torch.stack(seg_feats).cpu()
        
        # data_dict[f"{key}_{start_sec:.2f}-{end_sec:.2f}"] = {"seg_feats": seg_feats, "locations": torch.tensor(out['locations']), "boundaries": torch.tensor(out['boundaries']), "spf":spf}
        # hard boundaries from VAD: [0.31 0.55], [0.60 0.78], [0.79 1.03]
        # narrow boundaries obtained from my model: [0.31 0.33], [0.35 0.41], [0.45 0.57], [0.58 0.63], [0.65 0.77], [0.78 0.81]
        # need to first assign my boundaries to hard boundaries, if more than half of the length of mine boundary fall into the hard boundary, it is assign to that hard boundary
        # then adjust the start/end point of my boundary to hard boundary if it exceeds
        # then take median on adjacent boundaries of mine to get true boundaries
        hard_pointer = 0
        narrow_pointer = 0 # s17_01a_000338-000385
        hard_bs = [[int(item.split("_")[-1].split("-")[0]), int(item.split("_")[-1].split("-")[1])] for item in cur_match_keys] # [[31, 55], [60, 78], [79, 103]]
        assignment = {i:[] for i in range(len(hard_bs))}
        narrow_bs = [[int(item[0]*100), int(item[1]*100)] for item in out['boundaries']] # [[31, 33], [35, 41], [45, 57], [58, 63], [65, 63], [78, 81]]
        for narrow_pointer  in range(len(narrow_bs)):
            cur_narrow_b = narrow_bs[narrow_pointer]
            for hard_pointer in range(len(hard_bs)):
                cur_hard_b = hard_bs[hard_pointer]
                if cur_narrow_b[0] >= cur_hard_b[0] and cur_narrow_b[1] <= cur_hard_b[1]: # h_s  n_s n_e    h_e
                    assignment[hard_pointer].append(cur_narrow_b)
                    assigned = True
                elif cur_narrow_b[0] >= cur_hard_b[0] and cur_narrow_b[1] > cur_hard_b[1]:  # h_s  n_s    h_e n_e
                    if cur_hard_b[1] - cur_narrow_b[0] >= (cur_narrow_b[1] - cur_narrow_b[0])/2:
                        assignment[hard_pointer].append([cur_narrow_b[0], cur_hard_b[1]])
                        assigned = True
                elif cur_narrow_b[0] < cur_hard_b[0] and cur_narrow_b[1] <= cur_hard_b[1]:  # n_s h_s    n_e  h_e
                    if cur_narrow_b[1] - cur_hard_b[0] >= (cur_narrow_b[1] - cur_narrow_b[0])/2:
                        assignment[hard_pointer].append([cur_hard_b[0], cur_narrow_b[1]])
                        assigned = True
                else:
                    assigned = False
                if assigned:
                    break
        adjusted_assignment = {}
        for i in assignment:
            if len(assignment[i]) == 0:
                adjusted_assignment[i] = [[0, hard_bs[i][1] - hard_bs[i][0] - 1]]
                continue
            boundaries = assignment[i]
            adjusted_boundaries = [0]
            for l, r in zip(boundaries[:-1], boundaries[1:]):
                adjusted_boundaries.append(int((l[1] + r[0])/2) - boundaries[0][0] - 1)
            adjusted_boundaries.append(boundaries[-1][1] - boundaries[0][0] - 1)
            adjusted_assignment[i] = [[adjusted_boundaries[j], adjusted_boundaries[j+1]] for j in range(len(adjusted_boundaries)-1)]
            adjusted_assignment[i][-1] = [adjusted_assignment[i][-1][0], max(hard_bs[i][1] - hard_bs[i][0] - 1, adjusted_assignment[i][-1][1])]
        # print(adjusted_assignment)
        # assert False
        for i in adjusted_assignment:
            with open(os.path.join(out_dir, f"{cur_match_keys[i]}.txt"), "w") as f:
                for [s, e] in adjusted_assignment[i]:
                    f.write(f"{s} {e} 9999999\n")
print(f"saved the txt files at {out_dir}")         
                




# if args.segment_method == "forceAlign":
#     print(f"missing alignments: {missing_ali}")

# with open(os.path.join(save_root, 'data_dict.pkl'), "wb") as f:
#     pickle.dump(data_dict, f)
# print(f"save pickle data at {os.path.join(save_root, 'data_dict.pkl')}")


