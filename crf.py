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

class CNN_CRF(nn.Module):
    def __init__(self, dropout=0.4, model='mufold'):
        super(CNN_CRF, self).__init__()
        self.cnn = MUFold_ss(10, dropout) if model == 'mufold' else Inception_ResNet(10)
        self.crf = CRF()

    def cnn_features(self, features, lengths):
        output = self.cnn(features)
        mask = cuda_tensor_wrapper(sequence_mask(lengths).unsqueeze(-1))
        output = output * cuda_var_wrapper(mask)
        return output

    def forward_alg(self, features, labels, lengths):
        features = self.cnn_features(features, lengths)
        neg_log_likelihood = self.crf.neg_log_likelihood(features, labels, lengths)
        return neg_log_likelihood

    def forward(self, features, lengths):
        features = self.cnn_features(features, lengths)
        _, output = self.crf(features, lengths)
        return output

class CRF(nn.Module):
    def __init__(self, tagset_size=10):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size # 8 labels + START_TAG + STOP_TAG
        # transitions[i,j] = the score of transitioning from j to i. Enforcing no 
        # transition from STOP_TAG and to START_TAG
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[START_TAG, :] = -10000
        self.transitions.data[:, STOP_TAG] = -10000

    def _score_sequence(self, feats, labels):
        # Gives the score of a provided tag sequence
        score = cuda_var_wrapper(torch.Tensor([0]))
        labels = torch.cat([cuda_tensor_wrapper(torch.LongTensor([START_TAG])), labels.data])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[labels[i + 1], labels[i]] + feat[labels[i + 1]]
        score = score + self.transitions[STOP_TAG, labels[-1]]
        return score

    def _score_sequence_sequential(self, feats, labels, lengths):
        # feats: [batch_size x seq_len x tagset_size]
        # labels: [batch_size x seq_len]
        scores = []
        for feat, label, length in zip(feats, labels, lengths):
            feat = feat[:length, :]
            label = label[:length]
            score = self._score_sequence(feat, label)
            scores.append(score)
        scores = torch.cat(scores, 0)
        return scores

    def _forward_alg_sequential(self, feats, lengths):
        """Do the forward algorithm to compute the partition function
        feats: [sequence_length, batch_size, tagset_size]
        """
        init_alphas = torch.Tensor(feats.size()[2]).fill_(-10000.)
        init_alphas[START_TAG] = 0
        alphas = []
        for i, feat_batch in enumerate(feats):
            forward_var = cuda_var_wrapper(init_alphas)
            for j, feat in enumerate(feat_batch):
                if j >= lengths[i]:
                    break
                emit_score = feat.unsqueeze(1)
                tag_var = forward_var + self.transitions + emit_score
                forward_var = log_sum_exp(tag_var)
            terminal_var = (forward_var + self.transitions[STOP_TAG])
            alpha = log_sum_exp(terminal_var)
            alphas.append(alpha)
        alphas = torch.cat(alphas, 0)
        return alphas

    def _viterbi_decode_sequential(self, feats, lengths):
        """Given the nn extracted features [Seq_len x Batch_size x tagset_size], return the Viterbi
           Decoded score and most probable sequence prediction. Input feats is assumed to have
           dimension of 3.
        """
        max_length = max(lengths)
        tagset_size = feats.size()[2]
        init_vvars = torch.Tensor(tagset_size).fill_(-10000.)
        init_vvars[START_TAG] = 0

        path_scores = []
        best_paths = []
        for i, feat_batch in enumerate(feats):
            backpointers = []
            forward_var = cuda_var_wrapper(init_vvars)
            for j, feat in enumerate(feat_batch):
                if j >= lengths[i]:
                    break
                next_tag_var = forward_var + self.transitions
                viterbi_vars, best_tag_id = next_tag_var.max(1)
                forward_var = (viterbi_vars + feat)
                backpointers.append(best_tag_id)
            terminal_var = (forward_var + self.transitions[STOP_TAG])
            path_score, best_tag_id = terminal_var.max(0)

            best_path = [best_tag_id]
            for backpointer in reversed(backpointers):
                # backpointer: [tagset]
                best_tag_id = backpointer[best_tag_id.data]
                best_path.append(best_tag_id)
            start = best_path.pop()
            best_path = torch.cat(best_path[::-1], 0)
            padding = cuda_tensor_wrapper(torch.zeros(max_length - lengths[i]).long())
            best_path = torch.cat((best_path.data, padding), 0)
            path_scores.append(path_score)
            best_paths.append(best_path)
        path_scores = torch.cat(path_scores, 0)
        best_paths = torch.stack(best_paths, 0)
        return path_scores, best_paths

    def neg_log_likelihood(self, feats, labels, lengths, debug=False):
        forward_score = self._forward_alg_sequential(feats, lengths)
        gold_score = self._score_sequence_sequential(feats, labels, lengths)
        if not debug:
            return torch.mean(forward_score - gold_score)
        else:
            return forward_score, gold_score

    def forward(self, feats, lengths):
        score, tag_seq = self._viterbi_decode_sequential(feats, lengths)
        return score, tag_seq
