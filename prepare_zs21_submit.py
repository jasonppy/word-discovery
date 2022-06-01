import argparse
import json
import os
import time
import torch
import tqdm
import numpy as np
import pickle
from collections import defaultdict, Counter
print("\nI am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_dir", type=str, default="/scratch/cluster/pyp/exp_pyp/discovery/word_unit_discovery/disc-23/mean_0.9_7_forceAlign")
parser.add_argument("--out_dir", type=str, default="/data2/scratch/pyp/exp_pyp/zs2020/2017/track2")
parser.add_argument("--k", type=int, default=4096)
parser.add_argument("--run_length_encoding", action="store_true", default=False, help="if True, collapse all adjacent same code into one code; if False, use the original implementation, which, when calculate word2code_recall, it collapse all same code within the same word into one code. and when calculating code2word_precision, it doesn't do anything, so if a code appears 10 times (within the interval of a word), this are accounted as coappearing 10 times ")
parser.add_argument("--iou", action="store_true", default=False, help="wether or not evaluate the intersection over union, center of mass distance, center of mass being in segment percentage")
parser.add_argument("--max_n_utts", type=int, default=200000, help="total number of utterances to study, there are 25020 for SpokenCOCO, so if the number is bigger than that, means use all utterances")
parser.add_argument("--topk", type=int, default=30, help="show stats of the topk words in hisst plot")
parser.add_argument("--tolerance", type=float, default=0.02, help="tolerance of word boundary match")

args = parser.parse_args()


def cal_code_boundary(centroid, data, spk):
    feats= data["seg_feats"]
    seg_center_in_sec = data["locations"]
    boundaries = data['boundaries']
    spf = data['spf']
    distances = (torch.sum(feats**2, dim=1, keepdim=True) 
                    + torch.sum(centroid**2, dim=1).unsqueeze(0)
                    - 2 * torch.matmul(feats, centroid.t()))
    codes = torch.min(distances, dim=1)[1].tolist()
    adjusted_boundaries = [boundaries[0][0]]
    for l, r in zip(boundaries[:-1], boundaries[1:]):
        adjusted_boundaries.append((l[1] + r[0])/2)
    adjusted_boundaries.append(boundaries[-1][1])
    cur_code2seg = defaultdict(list)
    for i, code in enumerate(codes):
        cur_code2seg[f"Class {code}"].append(f"{spk} {adjusted_boundaries[i].item():.2f} {adjusted_boundaries[i+1]:.2f}")
    return cur_code2seg

# submission file content
# Class 0
# s0019 5839.17 5839.43
# s0107 3052.89 3053.17
# s0107 4657.09 4657.45
# s1724 5211.24 5211.59
# s1724 10852.39 10852.72
# s2544 4561.61 4561.9
# s2544 6186.02 6186.36
# s2544 8711.48 8711.73
# s3020 11256.47 11256.82
# s5157 459.55 459.86
# s5968 1359.01 1359.3

# Class 1
# s0107 6531.34 6531.63
# s4018 206.01 206.31
# s6519 547.35 547.69
# ...
# ...

def prepare_data(centroid, exp_dir):
    with open(os.path.join(exp_dir, "data_dict.pkl"), "rb") as f:
        data_dict = pickle.load(f)
    t0 = time.time()
    code2seg = defaultdict(list)
    for key in tqdm.tqdm(data_dict):
        spk = key.split("_")[0]
        cur_code2seg = cal_code_boundary(centroid, data_dict[key], spk)
        for class_name in cur_code2seg:
            code2seg[class_name] += cur_code2seg[class_name]
    return code2seg

kmeans_dir = f"{args.exp_dir}/kmeans_models/CLUS{args.k}/centroids.npy"
centroid = torch.from_numpy(np.load(kmeans_dir))

code2seg = prepare_data(centroid, args.exp_dir)

if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)
with open(os.path.join(args.out_dir, "english.txt"), "w") as f:
    i = 0
    for j, key in enumerate(code2seg):
        if len(code2seg[key]) == 0:
            continue
        f.write(key)
        for item in code2seg[key]:
            f.write(f"{item}\n")
        i += 1
        f.write("\n")
print(f"find {i} classes in total")