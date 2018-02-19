from __future__ import print_function, division
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from mufold import MUFold_ss
from resnet import Inception_ResNet
from crf import CRF
from utils import *


class CNN_CRF(nn.Module):
    def __init__(self, dropout=0.4, model='mufold', n_cnn=4):
        super(CNN_CRF, self).__init__()
        self.cnn = MUFold_ss(10, dropout) if model == 'mufold' else Inception_ResNet(10, n_cnn)
        self.crf = CRF()

    def cnn_features(self, features, lengths):
        output = self.cnn(features)
        mask = cuda_tensor_wrapper(sequence_mask(lengths).unsqueeze(-1))
        output = output * cuda_var_wrapper(mask)
        output = output.t()
        return output

    def forward_alg(self, features, labels, lengths):
        features = self.cnn_features(features, lengths)
        labels = labels.t()
        neg_log_likelihood = self.crf.neg_log_likelihood(features, labels, lengths)
        return neg_log_likelihood

    def forward(self, features, lengths):
        features = self.cnn_features(features, lengths)
        _, output = self.crf(features, lengths)
        return features, output


if __name__ == '__main__':
    model = CNN_CRF()
    for name, child in model.named_children():
        if isinstance(child, CRF):
            print(name)
            print(child.transitions)