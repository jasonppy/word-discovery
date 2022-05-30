# 1. Environment
It is recommended to create a new conda environment for this project with `conda create -n wd python=3.9`, the requirement on python version is not rigid, as long as you can install the packages listed in `./requirements.txt`. The requirement for the versions of packages is not rigid either, the listed version is tested, but lower version might also work.

If you want to get the attention weights of different attention head (**which is required for all word and boundary detection results**), you need to modify the output of the `multi_head_attention_forward` function in the PyTorch package at`torch/nn/functional`. if you install pytorch using conda in environment `wd`, the path of the file should be `path_to_conda/envs/wd/lib/python3.9/site-packages/torch/nn/functional.py`. get to function `multi_head_attention_forward`, and change the output as the following

```python
    # if need_weights:
    #     # average attention weights over heads
    #     attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    #     return attn_output, attn_output_weights.sum(dim=1) / num_heads
    # else:
    #     return attn_output, None
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    return attn_output, attn_output_weights
```

Simply put, originally, the return of `attn_output_weights` is summed over all attention heads, and we don't want to do that so that we can have the attention weights from different heads.

# 2. Apply VG-HuBERT on Speech Segmentation
To enable quickly applying the VG-HuBERT on speech segmentation, we provide the following standalone script. You need to provide four arguments to make it run:

1. `model_path`. It should be the directory the `.pth` and `args.pkl` are at. We open provide two trained models, [VG-HuBERT_3](link) and [VG-HuBERT_4](link). VG-HuBERT_3 should performance better on speech segmentation.

2. `wav_file`. The speech file you want to segment, we recommend the length of the speech to be 1 ~ 8 seconds, although in our experience the segmentation performance of VG-HuBERT is robust to the length of the input. the file should be [SoundFlie](https://pysoundfile.readthedocs.io/en/latest/) Readable, i.e. .wav, .flac etc.

3. `tgt_layer` and `threshold`. The former is the layer from which you want to use the attention, the later is the threshold on the attention weights, below which are set to 0. For VG-HuBERT_3, we recommend `(tgt_layer, threshold) = (9, 0.7)` and for VG-HuBERT_4, we recommend `(tgt_layer, threshold) = (9, 0.8)`. Note that layer is 0-based, i.e. layer number starts from 0.

```python
model_path = # TODO
wav_file = # TODO
tgt_layer = # TODO
threshold = # TODO

import torch
import soundfile as sf
from models import audio_encoder
from itertools import groupby
from operator import itemgetter

def cls_attn_seg(cls_attn_weights, threshold, spf):
    threshold_value = torch.quantile(cls_attn_weights, threshold, dim=-1, keepdim=True) # [n_h, T]
    cls_attn_weights_sum = cls_attn_weights.sum(0)
    boundary_idx = torch.where((cls_attn_weights >= threshold_value).float().sum(0) > 0)[0].cpu().numpy()
    attn_boundary_intervals = []
    word_boundaries_intervals = []
    for k, g in groupby(enumerate(boundary_idx), lambda ix : ix[0] - ix[1]):
        seg = list(map(itemgetter(1), g))
        t_s, t_e = seg[0], min(seg[-1]+1, cls_attn_weights.shape[-1])
        if len(seg) > 1:
            attn_boundary_intervals.append([spf*t_s, spf*t_e])
    # take the mid point of adjacent attn boundaries as the predicted word boundaries
    word_boundary_list = [attn_boundary_intervals[0][0]/2] # fist boundary
    for left, right in zip(attn_boundary_intervals[:-1], attn_boundary_intervals[1:]):
        word_boundaries_list.append((left[1]+right[0])/2.)
    word_boundaries_line.append((attn_boundary_intervals[-1][1]+threshold_value.shape[-1]*spf)/2) # last boundary
    for i in range(len(word_boundaries_list)-1):
        word_boundary_intervals.append([word_boundaries_line[i], word_boundaries_line[i+1]])
    return {"attn_boundary_intervals": attn_boundary_intervals, "word_boundary_intervals": word_boundary_intervals}

# setup model
with open(os.path.join(model_path, "args.pkl"), "rb") as f:
    model_args = pickle.load(f)
model = audio_encoder.AudioEncoder(model_args)
bundle = torch.load(os.path.join(model_path, "best_bundle.pth"))
model.carefully_load_state_dict(bundle['dual_encoder'], load_all=True)
model.eval()
model = model.cuda()

# load waveform (do not layer normalize the waveform!)
audio, sr = sf.read(wav_file, dtype = 'float32')
assert sr == 16000
audio = torch.from_numpy(audio).unsqueeze(0).cuda() # [T] -> [1, T]

# model forward
with torch.no_grad():
    model_out = model(audio, padding_mask=None, mask=False, need_attention_weights=True, tgt_layer=tgt_layer)
feats = model_out['features'].squeeze(0)[1:] # [1, T+1, D] -> [T, D]
spf = audio.shape[-1]/sr/feats.shape[-2]
attn_weights = model_out['attn_weights'].squeeze(0) # [1, num_heads, T+1, T+1] -> [num_heads, T+1, T+1] (for the two T+1, first is target length then the source)
cls_attn_weights = attn_weights[:, 0, 1:] # [num_heads, T+1, T+1] -> [num_heads, T]
out = cls_attn_seg(cls_attn_weights, threshold, spf) # out contains attn boundaries and word boundaries in intervals
```

# 3. Score Models on Buckeye Segmentation and ZeroSpeech2020
This section illustrate how evaluate VG-HuBERT on Buckeye and ZeroSpeech2020 (i.e. to get result in table 4 and 5 in our [word discovery paper](https://arxiv.org/pdf/2203.15081.pdf))

## 3.1 Buckeye
Buckeye evaluating largely utilize [Herman's repo](https://github.com/kamperh/vqwordseg), which accompanies his [recent paper](https://arxiv.org/abs/2202.11929). Thank Herman for publish the paper and codebase at the same time!

First of all, please create a folder for all the data you will download to later, let's name it `/Buckeye/`
Please follow the [original website](https://buckeyecorpus.osu.edu/) to obtain the Buckeye dataset, put all `s*` folders in `/Buckeye/`. After that, download the VAD file from [Herman repo](https://github.com/kamperh/zerospeech2021_baseline/tree/master/datasets/buckeye) and put them input `/Buckeye/vad/` folder. Lastly, download the ground truth word alignment file from [this link](https://github.com/kamperh/vqwordseg/releases/download/v1.0/buckeye.zip), extract all three folders and put them in `/Buckeye/buckeye_segment_alignment`. 

After downloading is done. change the `model_root` and `data_root` in `./scripts/single_run_buckeye.sh`, then

```bash
cd scripts
bash single_run_buckeye.sh vg-hubert_3 9 0.7 mean clsAttn buckeyetest
```
you should get
```bash
Word boundaries:
Precision: 47.56%
Recall: 42.34%
F-score: 44.80%
OS: -10.98%
R-value: 54.15%
----------------------------
Word token boundaries:
Precision: 32.26%
Recall: 29.81%
F-score: 30.99%
OS: -7.58%
```

## 3.2 ZeroSpeech2020
Let me take a break before starting this


# 4. Training Dataset
Feel free to skip this section if you don't want to train the model and don't want to test the word discovery performance on SpokenCOCO.

For training, we use SpokenCOCO, you can download the spoken captions at []() and download the images from [the MSCOCO website](https://cocodataset.org/#download) via
```bash
coco_root=path/to/coco
wget https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz -P ${coco_root} # 64G
wget http://images.cocodataset.org/zips/train2014.zip -P {coco_root}
wget http://images.cocodataset.org/zips/val2014.zip -P {coco_root}
```
Please untar/unzip the compressed files after downloading them

If you want to train the model on Places, please follow
```bash
# Images
# please follow http://places.csail.mit.edu/downloadData.html

# spoken captions (85G)
places_root=${data_root}/places
wget https://data.csail.mit.edu/placesaudio/placesaudio_2020_splits.tar.gz -P ${places_root}
cd ${places_root}
tar -xf placesaudio_2020_splits.tar.gz
```


