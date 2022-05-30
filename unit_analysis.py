# Author: Wei-Ning Hsu, Puyuan Peng
import argparse
import json
import os
import time
import torch
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

STOP_WORDS = [
    "<SPOKEN_NOISE>",
    "<UNK>"
]

def prepare_data(centroid, json_fn, max_n_utts, exp_dir, run_length_encoding=False, A=None, b=None):
    utt2codes = {}
    utt2words = {}
    tot_nframes = 0
    with open(json_fn,"r") as f:
        data_json = json.load(f)['data']
    n_utts = min(len(data_json), max_n_utts)
    with open(os.path.join(exp_dir, "data_dict.pkl"), "rb") as f:
        data_dict = pickle.load(f)
    t0 = time.time()
    for utt_index in range(n_utts):
        temp = get_word_ali(data_json, utt_index)
        if temp != None:
            utt2words[utt_index] = temp
            utt2codes[utt_index] = get_code_ali_selective(centroid, data_dict, data_json[utt_index]['caption']['wav'] if 'caption' in data_json[utt_index] else data_json[utt_index]['wav'], run_length_encoding, A, b)
            tot_nframes += len(utt2codes[utt_index])
    t1 = time.time()
    
    print('Took %.fs to dump %d code frames for %d utts'
          % (t1 - t0, tot_nframes, n_utts))
    return utt2codes, utt2words

def get_word_ali(json_data, index):
    """
    raw_ali is a string like 'start1__word1__end1 start2__word2__end2 ...'
    """
    raw_ali = json_data[index].get('text_alignment', None)
    if raw_ali is None:
        return None
    
    data = []
    meta_toks = raw_ali.split()
    for meta_tok in meta_toks:
        toks = meta_tok.split('__')
        if len(toks) == 3:
            data.append((float(toks[0]), float(toks[2]), toks[1]))
    
    if len(data) == 0:
        return None
    else:
        return SparseAlignment(data)


def get_code_ali_selective(centroid, data_dict, wav_id, run_length_encoding=False, A=None, b=None):
    item = data_dict[wav_id]
    feats= item["seg_feats"]
    seg_center_in_sec = item["locations"]
    boundaries = item['boundaries']
    spf = item['spf']
    if A != None and b != None:
        feats = feats @ A + b
    if centroid.shape[0] == 3: # this is for A40, as we didn't run FAISS on A40
        codes = np.random.randint(0,100,feats.shape[0]).tolist()
    else:
        distances = (torch.sum(feats**2, dim=1, keepdim=True) 
                        + torch.sum(centroid**2, dim=1).unsqueeze(0)
                        - 2 * torch.matmul(feats, centroid.t()))
        codes = torch.min(distances, dim=1)[1].tolist()
    # _, codes = centroid.search(feats.numpy(), 1)
    data = []
    for i, (code, center_in_sec, boundary) in enumerate(zip(codes, seg_center_in_sec, boundaries)):
        data.append((center_in_sec-spf/2., center_in_sec+spf/2., boundary[0].item(), boundary[1].item(), [code], feats[i]))
    
    if run_length_encoding:
        raise NotImplementedError("no need for run length encoding")

    return SparseAlignment_code(data)


def comp_code_to_wordprec(utt2codes, utt2words, stop_words):
    """
    Return
        code_to_wordprec (dict) : code_to_wordprec[code] is a list of (word,
            precision, num_occ) sorted by precision 
    """
    ts = time.time()
    code_to_nsegs = defaultdict(int)
    code_to_feats = defaultdict(list)
    code_to_wordcounts = defaultdict(Counter)
    for utt_index in utt2codes:
        code_ali = utt2codes[utt_index]    
        word_ali = utt2words[utt_index]
        if word_ali is None:
            continue
        for codes, seg_wordset, feat in align_word_and_code(word_ali, code_ali):
            for code in codes:
                code_to_nsegs[code] += 1 # denominator is code
                code_to_wordcounts[code].update(seg_wordset)
                code_to_feats[code].append(feat)

    code_to_wordprec = dict()     
    n_codes_with_no_words = 0
    for code, nsegs in sorted(code_to_nsegs.items()):
        word_counts = code_to_wordcounts[code]
        word_prec_list = [(word, float(occ)/nsegs, occ) for word, occ \
                          in word_counts.most_common() if word not in stop_words] # co-occur / code occur, this has the problem that if code appear a lot of times within the interval of a word, the resulting precision can be high, run length encoding cannot completely solve this problem as it only tackle adjacent same words, what if within a word, the codes is [12,23,12,44,12,102], in this case 12 con't be collapsed
        n_codes_with_no_words += int(not bool(word_prec_list))
        code_to_wordprec[code] = word_prec_list
    
    print('%d / %d codes mapped to only utterances with empty transcripts' % (
          n_codes_with_no_words, len(code_to_nsegs)))

    # calculate variance of each code cluster
    code_to_variance = dict()
    for code in code_to_feats:
        code_to_variance[code] = torch.stack(code_to_feats[code],dim=0).var(dim=0).mean()
    return code_to_wordprec, code_to_variance

def find_boundary_matches(gt, pred, tolerance):
    """
    gt: list of ground truth boundaries
    pred: list of predicted boundaries
    all in seconds
    """
    gt_pointer = 0
    pred_pointer = 0
    gt_len = len(gt)
    pred_len = len(pred)
    match_pred = 0
    match_gt = 0
    while gt_pointer < gt_len and pred_pointer < pred_len:
        if np.abs(gt[gt_pointer] - pred[pred_pointer]) <= tolerance:
            match_gt += 1
            match_pred += 1
            gt_pointer += 1
            pred_pointer += 1
        elif gt[gt_pointer] > pred[pred_pointer]:
            pred_pointer += 1
        else:
            gt_pointer += 1
    # this another way to calculating boundary metrics, the resulting number will be a little bit higher
    # for pred_i in pred:
    #     min_dist = np.abs(gt - pred_i).min()
    #     match_pred += (min_dist <= tolerance)
    # for y_i in gt:
    #     min_dist = np.abs(pred - y_i).min()
    #     match_gt += (min_dist <= tolerance)
    return match_gt, match_pred, gt_len, pred_len



def comp_word_to_coderecall(utt2codes, utt2words, target_words, tolerance):
    """
    Compute recall of given words. If `target_words == []`, compute all words.
    
    Return
        word_to_coderecall (dict) : word_to_coderecall[word] is a list of
            (code, recall, num_occ) sorted by recall
    """
    ts = time.time()
    word_to_nsegs = defaultdict(int)
    code_to_nsegs = Counter()
    code_to_feats = defaultdict(list)
    word_to_codecounts = defaultdict(Counter)
    missing_words = Counter()
    IoU = []
    IoT = []
    seg_in_word = []
    CenterDist = []
    CenterIn = []
    coverage_per_sent = []
    match_gt_count = 0
    match_pred_count = 0
    gt_b_len = 0
    pred_b_len = 0
    for utt_index in utt2codes:
        cur_missing = 0
        cur_words = 0
        code_ali = utt2codes[utt_index]
        word_ali = utt2words[utt_index]
        if word_ali is None:
            continue
        gt_boundaries = np.unique([[item[0], item[1]] for item in word_ali.data])
        pred_boundaries = []
        for l, r in zip(code_ali.data[:-1], code_ali.data[1:]):
            pred_boundaries.append((r[2] + l[3])/2)
        a, b, c, d = find_boundary_matches(gt_boundaries[1:-1], pred_boundaries, tolerance) # exclude the first and last boundary from GT boundaries, these are already excluded in pred_boundaries
        match_gt_count += a
        match_pred_count += b
        gt_b_len += c
        pred_b_len += d
        for word_s, word_e, word in word_ali.data:
            if target_words and (word not in target_words):
                continue
            if word in STOP_WORDS:
                continue
            cur_words += 1
            seg_codelist, codeseg_boundaries = code_ali.get_segment(word_s, word_e)
            word_to_nsegs[word] += 1 # even there is no code in this interval, an predefined empty code -1 will be assigned
            seg_codeset = set(seg_codelist)
            word_to_codecounts[word].update(seg_codeset)
            code_to_nsegs.update(seg_codeset)
            for _, _, _, _, code, feat in code_ali._data:
                code_to_feats[code[0]].append(feat) # code is stored as [code]
            cur_max = 0
            cur_s, cur_e = None, None
            for code, (s, e) in zip(seg_codelist, codeseg_boundaries):
                if code == -1: # empty code scenario
                    missing_words[word] += 1
                    cur_missing += 1
                    break
                if s >= word_s and e <= word_e:
                    cur_iou = (e - s) / (word_e - word_s)
                    cur_iot = cur_iou
                    seg_in_word.append(1.)
                elif s < word_s and e <= word_e:
                    if (e - word_s) < (word_s - s): # c_s     w_s c_e    w_e
                        continue 
                    cur_iou = (e - word_s) / (word_e - s)
                    cur_iot = (e - word_s) / (word_e - word_s)
                    seg_in_word.append(0.)
                elif s >= word_s and e > word_e:
                    if (word_e - s) < (e - word_e): # w_s      c_s  w_e         c_e
                        continue
                    cur_iou = (word_e - s) / (e - word_s)
                    cur_iot = (word_e - s) / (word_e - word_s)
                    seg_in_word.append(0.)
                elif s < word_s and e > word_e:
                    if (word_e - word_s) < (e - s): # c_s    w_s  w_e       c_e
                        continue
                    cur_iou = (word_e - word_s) / (e - s)
                    cur_iot = 1.
                    seg_in_word.append(0.)
                IoT.append(cur_iot)
                IoU.append(cur_iou)
                word_center = (word_e + word_s)/2.
                seg_center = (e+s)/2.
                CenterDist.append(np.abs(seg_center - word_center))
                CenterIn.append(1. if word_center >= s and word_center <= e else 0.)
        coverage_per_sent.append((cur_words - cur_missing)/cur_words)
    IoU = np.mean(IoU)
    IoT = np.mean(IoT)
    SegInWord = np.mean(seg_in_word)
    CenterDist = np.mean(CenterDist)
    CenterIn = np.mean(CenterIn)
    word_missing_percentage = {}
    total_words = 0
    total_uncovered_words = 0
    for word in word_to_nsegs:
        total_words += word_to_nsegs[word]
        if word in missing_words:
            total_uncovered_words += missing_words[word]
            word_missing_percentage[word] = missing_words[word] / word_to_nsegs[word]
        else:
            word_missing_percentage[word] = 0.
    code_coverage = 1 - total_uncovered_words / total_words
    # sns.histplot(coverage_per_sent)
    # plt.savefig("./coverage_hist.png")
    coverage_per_sent = np.mean(coverage_per_sent)

    word_to_coderecall = dict()
    for word, nsegs in word_to_nsegs.items():
        code_counts = word_to_codecounts[word]
        code_recall_list = [(code, float(occ)/nsegs, occ) for code, occ \
                            in code_counts.most_common() if code != -1] # (word,code) cooccur / word occ, make sure nominator is a fraction of word occ, within the interval of a word, only count the same code once
        word_to_coderecall[word] = code_recall_list

    code_to_wordprec = dict()
    for code, nsegs in code_to_nsegs.items():
        if code == -1:
            continue
        word_precision_list = [(word, float(word_to_codecounts[word][code])/nsegs, word_to_codecounts[word][code]) for word in word_to_codecounts if code in word_to_codecounts[word]]
        word_precision_list.sort(key=lambda x: -x[1]) # rank by co-occurance, but later the pairs are ranked by F1, so this doesn't really matter
        code_to_wordprec[code] = word_precision_list

    # calculate variance of each code cluster
    code_to_variance = dict()
    for code in code_to_feats:
        code_to_variance[code] = torch.stack(code_to_feats[code],dim=0).var(dim=0).mean()
    
    b_prec = match_pred_count / pred_b_len
    b_recall = match_gt_count / gt_b_len
    b_f1 = compute_f1(b_prec, b_recall)
    b_os = b_recall / b_prec - 1.
    b_r1 = np.sqrt((1-b_recall)**2 + b_os**2)
    b_r2 = (-b_os + b_recall - 1) / np.sqrt(2)
    b_r_val = 1. - (np.abs(b_r1) + np.abs(b_r2))/2.
    return word_to_coderecall, code_to_wordprec, code_to_variance, word_to_nsegs, missing_words, word_missing_percentage, code_coverage, coverage_per_sent, IoU, IoT, SegInWord, CenterDist, CenterIn, b_prec, b_recall, b_f1, b_os, b_r_val

def compute_f1(prec, recall):
    return 2*prec*recall / (prec+recall)

def comp_code_word_f1(code_to_wordprec, word_to_coderecall, min_occ):
    """
    Returns:
        code_to_wordf1 (dict) : code maps to a list of (word, f1, prec, recall, occ)
        word_to_codef1 (dict) : word maps to a list of (code, f1, prec, recall, occ)
    """
    code_to_word2prec = {}
    for code in code_to_wordprec:
        wordprec = code_to_wordprec[code]
        code_to_word2prec[code] = {word : prec for word, prec, _ in wordprec}

    word_to_code2recall = {}
    for word in word_to_coderecall:
        coderecall = word_to_coderecall[word]
        word_to_code2recall[word] = {code : (recall, occ) \
                                     for code, recall, occ in coderecall}

    code_to_wordf1 = defaultdict(list)
    for code in code_to_word2prec:
        for word, prec in code_to_word2prec[code].items():
            recall, occ = word_to_code2recall.get(word, {}).get(code, (0, 0))
            if occ >= min_occ:
                f1 = compute_f1(prec, recall)
                code_to_wordf1[code].append((word, f1, prec, recall, occ))
        code_to_wordf1[code] = sorted(code_to_wordf1[code], key=lambda x: -x[1])

    word_to_codef1 = defaultdict(list)
    for word in word_to_code2recall:
        for code, (recall, occ) in word_to_code2recall[word].items():
            if occ >= min_occ:
                prec = code_to_word2prec.get(code, {}).get(word, 0)
                f1 = compute_f1(prec, recall)
                word_to_codef1[word].append((code, f1, prec, recall, occ))
        word_to_codef1[word] = sorted(word_to_codef1[word], key=lambda x: -x[1])

    return code_to_wordf1, word_to_codef1


########################################################################################
def compute_f1(prec, recall):
    return 2*prec*recall / (prec+recall)

def comp_code_word_f1(code_to_wordprec, word_to_coderecall, min_occ):
    """
    Returns:
        code_to_wordf1 (dict) : code maps to a list of (word, f1, prec, recall, occ)
        word_to_codef1 (dict) : word maps to a list of (code, f1, prec, recall, occ)
    """
    code_to_word2prec = {}
    for code in code_to_wordprec:
        wordprec = code_to_wordprec[code]
        code_to_word2prec[code] = {word : prec for word, prec, _ in wordprec}

    word_to_code2recall = {}
    for word in word_to_coderecall:
        coderecall = word_to_coderecall[word]
        word_to_code2recall[word] = {code : (recall, occ) \
                                     for code, recall, occ in coderecall}

    code_to_wordf1 = defaultdict(list)
    for code in code_to_word2prec:
        for word, prec in code_to_word2prec[code].items():
            recall, occ = word_to_code2recall.get(word, {}).get(code, (0, 0))
            if occ >= min_occ:
                f1 = compute_f1(prec, recall)
                code_to_wordf1[code].append((word, f1, prec, recall, occ))
        code_to_wordf1[code] = sorted(code_to_wordf1[code], key=lambda x: -x[1]) # rank by F1

    word_to_codef1 = defaultdict(list)
    for word in word_to_code2recall:
        for code, (recall, occ) in word_to_code2recall[word].items():
            if occ >= min_occ:
                prec = code_to_word2prec.get(code, {}).get(word, 0)
                f1 = compute_f1(prec, recall)
                word_to_codef1[word].append((code, f1, prec, recall, occ))
        word_to_codef1[word] = sorted(word_to_codef1[word], key=lambda x: -x[1]) # rank by F1

    return code_to_wordf1, word_to_codef1


########################################################################################
class Alignment(object):
    def __init__(self):
        raise

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self._data


class SparseAlignment(Alignment):
    """
    alignment is a list of (start_time, end_time, value) tuples.
    """
    def __init__(self, data, unit=1.):
        self._data = [(s*unit, e*unit, v) for s, e, v in data]

    def __repr__(self):
        return str(self._data)

    def get_segment(self, seg_s, seg_e, empty_word='<SIL>'):
        """
        return words in the given segment.
        """
        seg_ali = self.get_segment_ali(seg_s, seg_e, empty_word)
        return [word for _, _, word in seg_ali.data]

    def get_segment_ali(self, seg_s, seg_e, empty_word=None, contained=False):
        seg_data = []
        if contained:
            is_valid = lambda s, e: (s >= seg_s and e <= seg_e)
        else:
            is_valid = lambda s, e: (max(s, seg_s) < min(e, seg_e))

        for (word_s, word_e, word) in self.data:
            if is_valid(word_s, word_e):
                seg_data.append((word_s, word_e, word))
        if not seg_data and empty_word is not None:
            seg_data = [(seg_s, seg_e, empty_word)]
        return SparseAlignment(seg_data)

    def get_words(self):
        return {word for _, _, word in self.data}

    def has_words(self, check_words):
        """check_words is assumed to be a set"""
        assert(isinstance(check_words, set))
        return bool(self.get_words().intersection(check_words))

class SparseAlignment_code(Alignment):
    """
    alignment is a list of (start_time, end_time, value) tuples.
    """
    def __init__(self, data):
        # v is a list of codes
        # self._data = [(s*unit, e*unit, v) for s, e, v in data]
        self._data = [(s, e, b_s, b_e, v, feats) for s, e, b_s, b_e, v, feats in data] # s,e are start end for center frame, b_s and b_e are start end for the segment (all in seconds)

    def __repr__(self):
        return str(self._data)

    def get_segment(self, seg_s, seg_e, empty_code=[-1]):
        """
        return codes in the given segment.
        """
        seg_ali = self.get_segment_ali(seg_s, seg_e, empty_code)
        return [codes[0] for _, _, _, _, codes, _ in seg_ali.data], [(codeseg_s, codeseg_e) for _, _, codeseg_s, codeseg_e, _, _ in seg_ali.data]

    def get_segment_ali(self, seg_s, seg_e, empty_code=[-1], contained=False):
        seg_data = []
        if contained:
            is_valid = lambda s, e: (s >= seg_s and e <= seg_e)
        else:
            is_valid = lambda s, e: (max(s, seg_s) < min(e, seg_e))

        
        for (code_s, code_e, codeseg_s, codeseg_e, code, feats) in self.data:
            if is_valid(code_s, code_e):
                seg_data.append((code_s, code_e, codeseg_s, codeseg_e, code, feats)) # code can repeat, this will lead to problems
        if not seg_data and empty_code is not None:
            seg_data = [(seg_s, seg_e, seg_s, seg_e, empty_code, feats)]
        return SparseAlignment_code(seg_data)

    def get_codes(self):
        raise NotImplementedError
        return {code for _, _, code in self.data}

    def has_codes(self, check_codes):
        """check_codes is assumed to be a set"""
        raise NotImplementedError
        assert(isinstance(check_codes, set))
        return bool(self.get_codes().intersection(check_codes))

class DenseAlignment(Alignment):
    """
    alignment is a list of values that is assumed to have equal duration.
    """
    def __init__(self, data, spf, offset=0):
        assert(offset >= 0)
        self._data = data
        self._spf = spf
        self._offset = offset
    
    @property
    def spf(self):
        return self._spf

    @property
    def offset(self):
        return self._offset

    def __repr__(self):
        return 'offset=%s, second-per-frame=%s, data=%s' % (self._offset, self._spf, self._data)

    def get_center(self, frm_index):
        return (frm_index + 0.5) * self.spf + self.offset

    def get_segment(self, seg_s, seg_e):
        """
        return words in the given segment
        """
        seg_frm_s = (seg_s - self.offset) / self.spf
        seg_frm_s = int(max(np.floor(seg_frm_s), 0))

        seg_frm_e = (seg_e - self.offset) / self.spf
        seg_frm_e = int(min(np.ceil(seg_frm_e), len(self.data)))
        
        seg_words = self.data[seg_frm_s:seg_frm_e]
        return seg_words

    def get_ali_and_center(self):
        """return a list of (code, center_time_sec)"""
        return [(v, self.get_center(f)) \
                for f, v in enumerate(self.data)]

    def get_sparse_ali(self):
        new_data = list(self.data) + [-1]
        changepoints = [j for j in range(1, len(new_data)) \
                        if new_data[j] != new_data[j-1]]
    
        prev_cp = 0
        sparse_data = []
        for cp in changepoints:
            t_s = prev_cp * self._spf + self._offset
            t_e = cp * self._spf + self._offset
            sparse_data.append((t_s, t_e, new_data[prev_cp]))
            prev_cp = cp
        return SparseAlignment(sparse_data)


##############################
# Transcript Post-Processing #
##############################

def align_sparse_to_dense(sp_ali, dn_ali, center_to_range):
    """
    ARGS:
        sp_ali (SparseAlignment):
        dn_ali (DenseAlignment):
    """
    ret = []
    w_s_list, w_e_list, w_list = zip(*sp_ali.data)
    w_sidx = 0  # first word that the current segment's start is before a word's end
    w_eidx = 0  # first word that the current segment's end is before a word's start
    for code, cs in dn_ali.get_ali_and_center():
        ss, es = center_to_range(cs)
        while w_sidx < len(w_list) and ss > w_e_list[w_sidx]:
            w_sidx += 1
        while w_eidx < len(w_list) and es > w_s_list[w_eidx]:
            w_eidx += 1
        seg_wordset = set(w_list[w_sidx:w_eidx]) if w_eidx > w_sidx else {'<SIL>'}
        ret.append((code, seg_wordset))
    return ret

def align_word_and_code(word_ali, code_ali):
    """
    ARGS:
        word_ali (SparseAlignment):
        code_ali (SparseAlignment):
    """
    ret = []
    w_s_list, w_e_list, w_list = zip(*word_ali.data)
    # c_s_list, c_e_list, c_list = zip(*code_ali.data)
    c_s_list, c_e_list, c_list, f_list = zip(*code_ali.data)
    w_sidx = 0  # first word that the current segment's start is before a word's end
    w_eidx = 0  # first word that the current segment's end is before a word's start
    for ss, es, code, feat in zip(c_s_list, c_e_list, c_list, f_list):
        while w_sidx < len(w_list) and ss > w_e_list[w_sidx]:
            w_sidx += 1
        while w_eidx < len(w_list) and es > w_s_list[w_eidx]:
            w_eidx += 1
        seg_wordset = set(w_list[w_sidx:w_eidx]) if w_eidx > w_sidx else {'<SIL>'}
        # ret.append((code, seg_wordset))
        ret.append((code, seg_wordset, feat))
    return ret

def print_code_to_word_prec(code_to_wordprec, prec_threshold=0.35,
                            num_show=3, show_all=True):
    n_codes = len(code_to_wordprec.keys())
    n_codes_above_prec_threshold = 0
            
    for code in sorted(code_to_wordprec.keys()):
        wordprec = code_to_wordprec[code]
        if not len(wordprec):
            continue
        (top_word, top_prec, _) = wordprec[0]
        above_prec_threshold = (top_prec >= prec_threshold)
        if above_prec_threshold:
            n_codes_above_prec_threshold += 1
            
        if show_all or above_prec_threshold:
            tot_occ = sum([occ for _, _, occ in wordprec])
            # show top-k
            msg = "%s %4d (#words=%5d, occ=%5d): " % (
                "*" if above_prec_threshold else " ", 
                code, len(wordprec), tot_occ)
            for word, prec, _ in wordprec[:num_show]:
                res = "%s (%5.2f)" % (word, prec)
                msg += " %-25s|" % res
            print(msg)
    
    print(('Found %d / %d (%.2f%%) codes with a word detector with' 
           'prec greater than %f.') % (
            n_codes_above_prec_threshold, n_codes, 
            n_codes_above_prec_threshold / n_codes * 100, prec_threshold)) 

def print_word_by_code_recall(word_to_coderecall, num_show=3):
    for word in sorted(word_to_coderecall):
        tot_occ = sum([o for _, _, o in word_to_coderecall[word]])
        print("%-15s (#occ = %4d)" % (word, tot_occ),
              [('%4d' % c, '%.2f' % r) for c, r, _ in word_to_coderecall[word][:num_show]])


def print_code_stats_by_f1(code_to_wordf1, code_ranks_show=range(10),
                           num_word_show=2):
    print("##### Showing ranks %s" % str(code_ranks_show))
    codes = sorted(code_to_wordf1.keys(), 
                   key=lambda x: (-code_to_wordf1[x][0][1] if len(code_to_wordf1[x]) else 0))
    print('%3s & %4s & %10s & %6s & %6s & %6s & %5s'
          % ('rk', 'code', 'word', 'F1', 'Prec', 'Recall', 'Occ'))
    for rank in code_ranks_show:
        if rank >= len(codes):
            continue
        code = codes[rank]
        msg = '%3d & %4d' % (rank+1, code)
        for word, f1, prec, recall, occ in code_to_wordf1[code][:num_word_show]:
            msg += ' & %10s & %6.2f & %6.2f & %6.2f & %5d' % (
                    word.lower(), f1*100, prec*100, recall*100, occ)
        msg += ' \\\\'
        print(msg)

def print_clustering_purity_nmi(code_to_word_f1):
    '''
    code_to_word_f1 is a dict, key is code, value is a list of (word, f1, prec, recall, occ), sorted by f1
    this method assign each cluster to words that appears the most (by occ)
    while the table ranks code by f1 (so assign code to word with the highest f1)
    '''
    from scipy import stats
    all_codes_count = {}
    all_words_count = defaultdict(int)
    cond_entropy = {}
    purity_numerator_occ = 0
    purity_numerator_f1 = 0
    for code in code_to_word_f1:
        if len(code_to_word_f1[code]) == 0:
            continue
        code_count = sum([item[-1] for item in code_to_word_f1[code]])
        all_codes_count[code] = code_count
        cond_prob = [item[-1]/code_count for item in code_to_word_f1[code]]
        cond_entropy[code] = stats.entropy(cond_prob)
        purity_numerator_occ += max([item[-1] for item in code_to_word_f1[code]]) # occ the most in this cluster
        purity_numerator_f1 += code_to_word_f1[code][0][-1] # has the highest f1 in this cluster
        for item in code_to_word_f1[code]:
            word = item[0]
            occ = item[-1]
            all_words_count[word] += occ
    num_total_codes_occ = sum([all_codes_count[code] for code in all_codes_count])
    print(f"number of total codes occurance: {num_total_codes_occ}")
    purity_occ = purity_numerator_occ / num_total_codes_occ
    purity_f1 = purity_numerator_f1 / num_total_codes_occ
    nmi = 2*sum([all_codes_count[code]/num_total_codes_occ * cond_entropy[code] for code in all_codes_count]) / (stats.entropy([all_codes_count[code] for code in all_codes_count]) + stats.entropy([all_words_count[word] for word in all_words_count]))
    print(f"purity (using most occured word as code assignment): {purity_occ:.3f}")
    print(f"purity (using highest f1 word as code assignment): {purity_f1:.3f}")
    print(f"nmi: {nmi:.3f}")

def plot_and_save(c2wf, c2var, basename):
    """
    c2wf, or code to word precision 
     is a dict, key is code, value is a list of (word, f1, prec, recall, occ), sorted by f1
    
    c2var, or code to cluster variance
     is a dict, key is code, value is mean variance of this cluster
    """
    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib
    matplotlib.style.use("ggplot")
    os.makedirs(f"../pics/{basename}", exist_ok=True)
    var_f1 = np.array([[c2var[code].item(), c2wf[code][0][1]] for code in c2wf if len(c2wf[code]) > 0])
    plt.figure()
    plt.xlabel("var")
    plt.ylabel("f1")
    plot = sns.scatterplot(x = var_f1[:,0], gt = var_f1[:,1])
    plot.figure.savefig(f"../pics/{basename}/var_f1.png")
    

    plt.figure()
    var_occ = np.array([[c2var[code].item(), c2wf[code][0][-1]] for code in c2wf if len(c2wf[code]) > 0])
    plt.xlabel("var")
    plt.ylabel("occ")
    plot = sns.scatterplot(x = var_occ[:,0], gt = var_occ[:,1])
    plot.figure.savefig(f"../pics/{basename}/var_occ.png")


    plt.figure()
    var_prec = np.array([[c2var[code].item(), c2wf[code][0][2]] for code in c2wf if len(c2wf[code]) > 0])
    plt.xlabel("var")
    plt.ylabel("prec")
    plot = sns.scatterplot(x = var_prec[:,0], gt = var_prec[:,1])
    plot.figure.savefig(f"../pics/{basename}/var_prec.png")

    plt.figure()
    var_recall = np.array([[c2var[code].item(), c2wf[code][0][3]] for code in c2wf if len(c2wf[code]) > 0])
    plt.xlabel("var")
    plt.ylabel("recall")
    plot = sns.scatterplot(x = var_recall[:,0], gt = var_recall[:,1])
    plot.figure.savefig(f"../pics/{basename}/var_recall.png")

def plot_missing_words(word_to_nsegs, missing_words, word_missing_percentage, topk, basename):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib
    matplotlib.style.use("ggplot")
    os.makedirs(f"../pics/{basename}", exist_ok=True)
    word_freq = sorted(word_to_nsegs.items(), key=lambda x: x[1], reverse=True)
    missing_freq = sorted(missing_words.items(), key=lambda x: x[1], reverse=True)
    percentage = sorted(word_missing_percentage.items(), key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(24,12))
    plt.title("Word Occurance")
    plt.ylabel("Counts")
    temp = word_freq[:topk]
    plt.xticks(range(len(temp)), [item[0] for item in temp])
    plt.bar(range(len(temp)), [item[1] for item in temp])
    plt.savefig(f"../pics/{basename}/word_occurance.png")
    
    plt.figure(figsize=(24,12))
    plt.title("Word Missing Count")
    plt.ylabel("Counts")
    temp = missing_freq[:topk]
    plt.xticks(range(len(temp)), [item[0] for item in temp])
    plt.bar(range(len(temp)), [item[1] for item in temp])
    plt.savefig(f"../pics/{basename}/word_missing.png")
    
    plt.figure(figsize=(24,12))
    plt.title("Word Missing Percentage")
    plt.ylabel("Percentage %")
    temp = percentage[:topk]
    plt.xticks(range(len(temp)), [item[0] for item in temp])
    plt.bar(range(len(temp)), [item[1] for item in temp])
    plt.savefig(f"../pics/{basename}/word_missing_percentage.png")

def threshold_by_var(c2wf, c2var, threshold):
    c2wf_th = dict()
    for code in c2var:
        if c2var[code] <= threshold:
            c2wf_th[code] = c2wf[code]
    return c2wf_th



def print_word_stats_by_f1(word_to_codef1, word_ranks_show=range(10),
                           num_code_show=3):
    print("##### Showing ranks %s" % str(word_ranks_show))
    words = sorted(word_to_codef1.keys(), 
                   key=lambda x: (-word_to_codef1[x][0][1] if len(word_to_codef1[x]) else 0))
    print('%3s & %15s & %4s & %6s & %6s & %6s & %5s'
          % ('rk', 'word', 'code', 'F1', 'Prec', 'Recall', 'Occ'))
    for rank in word_ranks_show:
        if rank >= len(words):
            continue
        word = words[rank]
        msg = '%3d & %15s' % (rank+1, word.lower())
        for code, f1, prec, recall, occ in word_to_codef1[word][:num_code_show]:
            # print("code: ", code)
            # print("occ: ", occ)
            msg += ' & %4d & %6.2f & %6.2f & %6.2f & %5d' % (
                    code, f1*100, prec*100, recall*100, occ)
        msg += ' \\\\'
        print(msg)
    

def count_high_f1_words(word_to_codef1, f1_threshold=0.5, verbose=True):
    count = 0 
    for word in word_to_codef1.keys():
        if len(word_to_codef1[word]) and (word_to_codef1[word][0][1] >= f1_threshold):
            count += 1
    if verbose:
        print('%d / %d words with an F1 score >= %s'
              % (count, len(word_to_codef1), f1_threshold))
    return count
    

def compute_topk_avg_f1(word_to_codef1, k=250, verbose=True):
    f1s = [word_to_codef1[word][0][1] for word in word_to_codef1 \
           if len(word_to_codef1[word])]
    top_f1s = sorted(f1s, reverse=True)[:k]
    
    if verbose:
        print('avg F1 = %.2f%% for top %d words; %.2f%% for all %d words'
              % (100*np.mean(top_f1s), len(top_f1s), 100*np.mean(f1s), len(f1s)))
    return 100*np.mean(top_f1s)


def print_analysis(code_to_wordprec, word_to_coderecall,
                   code_norms, high_prec_words, min_occ, rank_range):
    print_code_to_word_prec(code_to_wordprec, prec_threshold=0.35,
                            num_show=3, show_all=True)
    print_word_by_code_recall(word_to_coderecall, num_show=3)

    code_to_wordf1, word_to_codef1 = comp_code_word_f1(
            code_to_wordprec, word_to_coderecall, min_occ=min_occ)
    print_code_stats_by_f1(code_to_wordf1, rank_range)
    print_word_stats_by_f1(word_to_codef1, rank_range)
    count_high_f1_words(word_to_codef1, f1_threshold=0.5)
    compute_topk_avg_f1(word_to_codef1, k=250)




def get_detection_stats(centroid, max_n_utts, exp_dir, json_fn, tolerance, stop_words=STOP_WORDS, min_occ=1, run_length_encoding=False, A=None, b=None):
    utt2codes, utt2words= prepare_data(centroid, json_fn, max_n_utts, exp_dir, run_length_encoding = run_length_encoding, A=A, b=b)
    w2cr, c2wp, c2var, word_to_nsegs, missing_words, word_missing_percentage, code_coverage, coverage_per_sent, IoU, IoT, SegInWord, CenterDist, CenterIn, b_prec, b_recall, b_f1, b_os, b_r_val = comp_word_to_coderecall(utt2codes, utt2words, [], tolerance)
    
    c2wf, w2cf = comp_code_word_f1(c2wp, w2cr, min_occ=min_occ)
    return c2wp, c2var, w2cr, c2wf, w2cf, word_to_nsegs, missing_words, word_missing_percentage, code_coverage, coverage_per_sent, IoU, IoT, SegInWord, CenterDist, CenterIn, b_prec, b_recall, b_f1, b_os, b_r_val
    
print("\nI am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_json", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json", help="the fn of force alignment json file")
parser.add_argument("--exp_dir", type=str, default="/scratch/cluster/pyp/exp_pyp/discovery/word_unit_discovery/disc-23/curFeats_mean_0.9_7_forceAlign")
parser.add_argument("--k", type=int, default=4096)
parser.add_argument("--run_length_encoding", action="store_true", default=False, help="if True, collapse all adjacent same code into one code; if False, use the original implementation, which, when calculate word2code_recall, it collapse all same code within the same word into one code. and when calculating code2word_precision, it doesn't do anything, so if a code appears 10 times (within the interval of a word), this are accounted as coappearing 10 times ")
parser.add_argument("--iou", action="store_true", default=False, help="wether or not evaluate the intersection over union, center of mass distance, center of mass being in segment percentage")
parser.add_argument("--max_n_utts", type=int, default=200000, help="total number of utterances to study, there are 25020 for SpokenCOCO, so if the number is bigger than that, means use all utterances")
parser.add_argument("--topk", type=int, default=30, help="show stats of the topk words in hisst plot")
parser.add_argument("--tolerance", type=float, default=0.02, help="tolerance of word boundary match")

args = parser.parse_args()
kmeans_dir = f"{args.exp_dir}/kmeans_models/CLUS{args.k}/centroids.npy"
centroid = torch.from_numpy(np.load(kmeans_dir))
A = None
b = None
c2wp, c2var, w2cr, c2wf, w2cf, word_to_nsegs, missing_words, word_missing_percentage, code_coverage, coverage_per_sent, IoU, IoT, SegInWord, CenterDist, CenterIn, b_prec, b_recall, b_f1, b_os, b_r_val = get_detection_stats(centroid, args.max_n_utts, tolerance= args.tolerance, exp_dir=args.exp_dir, json_fn=args.data_json, min_occ=1, run_length_encoding=args.run_length_encoding, A=A, b=b)
print(f"AScore: {2./(1/IoU+1/code_coverage):.4f}")
print(f"IoU: {IoU:.4f}")
print(f"IoT: {IoT:.4f}")
print(f"Percentage that the segment falls within the word interval: {SegInWord*100:.2f}")
print(f"Average distance (in seconds) between segment center and word center: {CenterDist:.4f}")
print(f"Percentage that word centers fall into the code segment: {CenterIn*100:.2f}%")
print(f"code coverage (average over all *words*): {code_coverage:.4f}")
print(f"code coverage (average over all *word types*): {sum([1-item for item in word_missing_percentage.values()])/len(word_missing_percentage):.4f}")
print(f"coverage per sentence: {coverage_per_sent:.4f}")
print(f"boundary precision: {b_prec:.4f}")
print(f"boundary recall: {b_recall:.4f}")
print(f"boundary F1: {b_f1:.4f}")
print(f"boundary over-segmentation: {b_os:.4f}")
print(f"boundary R value: {b_r_val:.4f}")
c2wf_th = threshold_by_var(c2wf, c2var, threshold=1)
print_clustering_purity_nmi(c2wf_th)

count_high_f1_words(w2cf, f1_threshold=0.5)
top_word_avg_f1 = compute_topk_avg_f1(w2cf, k=250)
print("done",flush=True)