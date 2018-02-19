from __future__ import print_function, division
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_loading import Protein_Dataset
from mufold import MUFold_ss
from resnet import Inception_ResNet
from utils import *


use_cuda = torch.cuda.is_available()
START_TAG = 8
STOP_TAG = 9


class CRF(nn.Module):
    def __init__(self, tagset_size=10):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size # 8 labels + START_TAG + STOP_TAG
        # transitions[i,j] = the score of transitioning from j to i. Enforcing no 
        # transition from STOP_TAG and to START_TAG
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[START_TAG, :] = -10000
        self.transitions.data[:, STOP_TAG] = -10000

    def _score_sequence(self, feats, labels, lengths):
        """
        args:
            feats (seq_len, batch_size, tagset_size): input score from previous layers
            labels (seq_len, batch_size): ground truth labels
            lengths (batch_size): length for each sequence
        return:
            score (batch_size): 

        """
        seq_len, batch_size, tagset_size = feats.size()
        scores = cuda_var_wrapper(torch.zeros(seq_len + 1, batch_size))
        start_labels = cuda_var_wrapper(torch.LongTensor(1, batch_size).fill_(START_TAG))
        stop_labels = cuda_var_wrapper(torch.LongTensor(batch_size).fill_(STOP_TAG))
        labels = torch.cat((start_labels, labels), 0)
        for i, feat in enumerate(feats):
            emit = feat.gather(1, labels[i + 1].unsqueeze(1)).squeeze()
            trans = self.transitions[labels[i + 1], labels[i]]
            scores[i + 1, :] = scores[i, :] + trans + emit
        last_score = scores.gather(0, lengths.unsqueeze(0))
        last_label = labels.gather(0, lengths.unsqueeze(0))
        score = last_score + self.transitions[stop_labels, last_label]
        score = score.squeeze()
        return score

    def _forward_alg(self, feats, lengths):
        seq_len, batch_size, tagset_size = feats.size()
        forward_var = cuda_var_wrapper(torch.Tensor(seq_len + 1, batch_size, tagset_size).fill_(-10000.))
        forward_var[..., START_TAG] = 0
        for i, feat in enumerate(feats):
            tag_var = forward_var[i, ...].unsqueeze(1) + self.transitions + feat.unsqueeze(-1)
            forward_var[i + 1, ...] = log_sum_exp(tag_var)
        idx = cuda_tensor_wrapper(torch.arange(batch_size).long())
        last_forward = forward_var.index_select(0, lengths)
        last_forward = last_forward[idx, idx, :]
        terminal_var = last_forward + self.transitions[STOP_TAG]
        alphas = log_sum_exp(terminal_var)
        return alphas

    def _viterbi_decode(self, feats, lengths):
        seq_len, batch_size, tagset_size = feats.size()
        forward_var = cuda_var_wrapper(torch.Tensor(seq_len + 1, batch_size, tagset_size).fill_(-10000.))
        best_tag_ids = torch.zeros(seq_len, batch_size, tagset_size).long()
        forward_var[..., START_TAG] = 0
        for i, feat in enumerate(feats):
            next_tag_var = forward_var[i, ...].unsqueeze(1) + self.transitions
            viterbi_vars, best_tag_id = next_tag_var.max(2)
            forward_var[i + 1, ...] = viterbi_vars + feat
            best_tag_ids[i, ...] = best_tag_id.data
        idx = cuda_tensor_wrapper(torch.arange(batch_size).long())
        last_forward = forward_var.index_select(0, lengths)
        last_forward = last_forward[idx, idx, :]
        terminal_var = last_forward + self.transitions[STOP_TAG]
        path_score, best_tag_id_last = terminal_var.max(1)
        best_paths = cuda_tensor_wrapper(torch.zeros(seq_len, batch_size).long())
        for i in range(batch_size):
            effective_len = lengths.data[i]
            effective_path = best_tag_ids[:effective_len, i, :]
            best_paths[effective_len - 1, i] = best_tag_id_last.data[i]
            for j in torch.arange(effective_len - 1, 0, -1).long():
                best_paths[j - 1, i] = effective_path[j, best_paths[j, i]]
        best_paths = best_paths.t()
        return path_score, best_paths

    def neg_log_likelihood(self, feats, labels, lengths, debug=False):
        gold_score = self._score_sequence(feats, labels, lengths)
        forward_score = self._forward_alg(feats, lengths)
        if not debug:
            return torch.mean(forward_score - gold_score)
        else:
            return forward_score, gold_score

    def forward(self, feats, lengths):
        score, tag_seq = self._viterbi_decode(feats, lengths)
        return score, tag_seq
