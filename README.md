# Word Discovery in Visually Grounded, Self-Supervised Speech Models
This is the codebase for [paper](https://arxiv.org/pdf/2203.15081.pdf) 
```
@inproceedings{peng2022word,
  title={Word Discovery in Visually Grounded, Self-Supervised Speech Models},
  author={Peng, Puyuan and Harwath, David},
  booktitle={Proc. INTERSPEECH},
  year={2022}
}
```

## 1. Environment
It is recommended to create a new conda environment for this project with `conda create -n wd python=3.9`, the requirement on python version is not rigid, as long as you can install the packages listed in `./requirements.txt`. The requirement for the versions of packages is not rigid either, the listed version is tested, but lower version might also work.

If you want to get the attention weights of different attention head (**which is required for all word and boundary detection experiments**), you need to modify the output of the `multi_head_attention_forward` function in the PyTorch package at`torch/nn/functional`. if you install pytorch using conda in environment `wd`, the path of the file should be `path_to_conda/envs/wd/lib/python3.9/site-packages/torch/nn/functional.py`. get to function `multi_head_attention_forward`, and change the output as the following

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

## 2. Apply VG-HuBERT on Speech Segmentation
To enable quickly applying the VG-HuBERT on speech segmentation, we provide the following standalone script. You need to provide four arguments to make it run:

1. `model_path`. It should be the directory the `.pth` and `args.pkl` are at. We open provide two trained models, [VG-HuBERT_3](https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar) and [VG-HuBERT_4](https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_4.tar). VG-HuBERT_3 should performance better on speech segmentation. Please untar the file after downloading.

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

def cls_attn_seg(cls_attn_weights, threshold, spf, audio_len_in_sec):
    threshold_value = torch.quantile(cls_attn_weights, threshold, dim=-1, keepdim=True) # [n_h, T]
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
    word_boundaries_line.append((attn_boundary_intervals[-1][1]+audio_len_in_sec)/2) # last boundary
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
audio_len_in_sec = len(audio) / sr
audio = torch.from_numpy(audio).unsqueeze(0).cuda() # [T] -> [1, T]

# model forward
with torch.no_grad():
    model_out = model(audio, padding_mask=None, mask=False, need_attention_weights=True, tgt_layer=tgt_layer)
feats = model_out['features'].squeeze(0)[1:] # [1, T+1, D] -> [T, D]
spf = audio.shape[-1]/sr/feats.shape[-2]
attn_weights = model_out['attn_weights'].squeeze(0) # [1, num_heads, T+1, T+1] -> [num_heads, T+1, T+1] (for the two T+1, first is target length then the source)
cls_attn_weights = attn_weights[:, 0, 1:] # [num_heads, T+1, T+1] -> [num_heads, T]
out = cls_attn_seg(cls_attn_weights, threshold, spf, audio_len_in_sec) # out contains attn boundaries and word boundaries in intervals
```
## 3. Speech Segmentation and Word Detection on SpokenCOCO
This section illustrates how to apply the VG-HuBERT model to segment speech and detect words in SpokenCOCO. Please first download the SpokenCOCO audios and MSCOCO images following:
```bash
coco_root=/path/to/coco/
wget https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz -P ${coco_root} # 64G
wget http://images.cocodataset.org/zips/train2014.zip -P {coco_root}
wget http://images.cocodataset.org/zips/val2014.zip -P {coco_root}
```
Please untar/unzip the compressed files after downloading them

Then download karpathy split json files with word alignment 
```bash
wget https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/karpathy_json.tar -P ${coco_root}/SpokenCOCO/
```
Please also untar it. 

Then you are all set, just run

```bash
cd ./scripts
bash run_spokencoco.sh vg-hubert_3 9 4096 0.7 max clsAttn 1 
```

after this please find `vg-hubert_3_spokencoco_max_0.7_9_clsAttn.log` in ./logs/ with the following content:
```log
AScore: 0.6591
IoU: 0.6151
IoT: 0.6603
Percentage that the segment falls within the word interval: 65.79
Average distance (in seconds) between segment center and word center: 0.0647
Percentage that word centers fall into the code segment: 85.47%
code coverage (average over all *words*): 0.7098
code coverage (average over all *word types*): 0.9623
coverage per sentence: 0.7095
boundary precision: 0.3619
boundary recall: 0.2721
boundary F1: 0.3107
boundary over-segmentation: -0.2480
boundary R value: 0.4459
number of total codes occurance: 179179
purity (using most occured word as code assignment): 0.759
purity (using highest f1 word as code assignment): 0.681
nmi: 0.115
1014 / 6085 words with an F1 score >= 0.5
avg F1 = 89.30% for top 250 words; 23.81% for all 6001 words
done
```

To get the note that we set different layers and threshold for getting the best Area group, Boundary group and Word group result (for def of the metrics in these groups, plz refer to the [paper](https://arxiv.org/pdf/2203.15081.pdf)). The hyperparameters and results on val set for VG-HuBERT_3 and VG-HuBERT_4 is...


## 4. Score VG-HuBERT on Buckeye Segmentation and ZeroSpeech2020
This section illustrate how evaluate VG-HuBERT on Buckeye and ZeroSpeech2020 (i.e. to get result in table 4 and 5 in our [word discovery paper](https://arxiv.org/pdf/2203.15081.pdf))

### 4.1 Buckeye
Buckeye evaluating largely utilize [Herman's repo](https://github.com/kamperh/vqwordseg), which accompanies his [recent paper](https://arxiv.org/abs/2202.11929). Thank Herman for publish the paper and codebase at the same time!

First of all, please create a folder for all the data you will download to later, let's name it `/Buckeye/`
Please follow the [original website](https://buckeyecorpus.osu.edu/) to obtain the Buckeye dataset, put all `s*` folders in `/Buckeye/`. After that, download the VAD file from [Herman repo](https://github.com/kamperh/zerospeech2021_baseline/tree/master/datasets/buckeye) and put them input `/Buckeye/vad/` folder. Lastly, download the ground truth word alignment file from [this link](https://github.com/kamperh/vqwordseg/releases/download/v1.0/buckeye.zip), extract all three folders and put them in `/Buckeye/buckeye_segment_alignment`. 

After downloading is done. change the `model_root` and `data_root` in `./scripts/run_buckeye.sh`, then

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

### 4.2 ZeroSpeech2020
We'll do the ZS2020 Spoken Term Discovery track (just English).

First follow the ZeroSpeech 2020 section on the ZeroSpeech website to download the data and ground truth labels (remember to also download `2017_vads.zip`). This should be free and easy, like the rest of the steps :). Suppose you have put the `2020` folder at `/zs20/`. Remember to put `ENGLISH_VAD.csv` from 2017_vads.zip at `/2020/ENGLISH_VAD.csv`.

Then install zerospeech 2020 evaluation toolkit following [this official repo](https://github.com/zerospeech/zerospeech2020). Assume you have clone the repo at `~/zerospeech2020`.

Now you should be ready to test the models on this task. similarly, change the `model_root` and `data_root` in `./scripts/run_zs20.sh` to the parent folder of your model folder and data folder (for data_root, is should be `/zs20` if you follow the above)

Then run
```bash
cd ./scripts
bash run_zs20.sh vg-hubert_3 9 16384 0.7 max clsAttn 1
```

The above run takes a long time, so better run it as a sbatch job, or open a tmux window for it. After the job finishes, you should be able to find the output file `~/zerospeech2020/english_vg-hubert_3_max_0.7_9_clsAttn_16384.json` with content:

```json
{
    "2017-track2": {
        "english": {
            "scores": {
                "ned": 0.42968707380581783,
                "coverage": 0.9552742717879052,
                "words": 93769
            },
            "details": {
                "boundary_precision": 0.4477182844906101,
                "boundary_recall": 0.5690987635775963,
                "boundary_fscore": 0.5011637494055812,
                "grouping_precision": "NA",
                "grouping_recall": "NA",
                "grouping_fscore": "NA",
                "token_precision": 0.1728015507130922,
                "type_precision": 0.06441361217459927,
                "token_recall": 0.16639803706534623,
                "type_recall": 0.28718143780905286,
                "token_fscore": 0.16953935014383406,
                "type_fscore": 0.1052255642372453,
                "words": 93769,
                "coverage": 0.9552742717879052,
                "ned": 0.42968707380581783,
                "pairs": 6122486
            }
        }
    }
}

```



## 5. Training
Feel free to skip this section if you don't want to train the model.

If you do want to train a VG-HuBERT model, please first download the weight of pretrained HuBERT and DINO-ViT, we use HuBERT Base trained on librispeech 960h unsup speech and DINO ViT Small 8x8,

We can download the weights via
```bash
pretrained_root=/path/to/pretrained/hubertAndDINO
wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt -P ${pretrained_root}
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth -P ${pretrained_root}
```

Then, Please follow section 3 to download SpokenCOCO dataset (don't forget the json file).

After that, change the `pretrained_root`, `model_root`, and `data_root` to desired dir, you are good to go

```bash
cd ./scripts
bash training.sh
```

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

