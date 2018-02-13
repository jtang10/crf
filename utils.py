from __future__ import print_function, division
import torch
import numpy as np
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    """ Given a vector, return log_sum_exp with last dimension reduced.
    """
    last_dimension = len(vec.size()) - 1
    max_score, _ = torch.max(vec, last_dimension)
    if last_dimension > 0:
        _exp = torch.exp(vec - max_score.unsqueeze(last_dimension))
        _sum = torch.sum(_exp, last_dimension)
    else:
        _exp = torch.exp(vec - max_score)
        _sum = torch.sum(_exp)
    _log = max_score + torch.log(_sum)
    return _log

def cuda_var_wrapper(var, volatile=False):
    """Use CUDA Variable if GPU available. Also specify volatile for inference.
    """ 
    var = Variable(var, volatile=volatile)
    if use_cuda:
        var = var.cuda()
    return var

def cuda_tensor_wrapper(tensor):
    if use_cuda:
        tensor = tensor.cuda()
    return tensor

def get_batch_accuracy(labels, output, lengths):
    correct = 0.0
    total = 0.0
    if len(output.size()) > 2:
        _, output = torch.max(output, 2)
    if isinstance(output, Variable):
        output = output.data
    correct_matrix = torch.eq(output, labels.data)
    for j, length in enumerate(lengths):
            correct += torch.sum(correct_matrix[j, :length])
    total += sum(lengths)
    return correct, total

def evaluate(model, dataloader, crf, criterion):
    model.eval()
    correct, total = 0, 0
    loss_total = 0
    for i, (features, labels, _, lengths) in enumerate(dataloader):
        max_length = max(lengths)
        features = cuda_var_wrapper(features[:, :, :max_length], volatile=True)
        labels = cuda_var_wrapper(labels[:, :max_length], volatile=True)
        if crf:
            output = model(features, lengths)
            loss = model.forward_alg(features, labels, lengths)
        else:
            output = model(features)
            loss = criterion(output.contiguous().view(-1, 8), labels.contiguous().view(-1))
        correct_batch, total_batch = get_batch_accuracy(labels, output, lengths)
        loss_total += loss.data[0] * features.size()[0]
        correct += correct_batch
        total += total_batch
    return correct / total, loss_total

def eval(model, features, labels, lengths, crf, criterion):
    model.eval()
    if crf:
        output = model(features, lengths)
        loss = model.forward_alg(features, labels, lengths)
    else:
        output = model(features)
        loss = criterion(output.contiguous().view(-1, 8), labels.contiguous().view(-1))
    loss_batch = loss.data[0] * features.size()[0]
    correct_batch, total_batch = get_batch_accuracy(labels, output, lengths)
    return correct_batch, total_batch, loss_batch

def sequence_mask(sequence_length, max_len=None):
    if isinstance(sequence_length, Variable):
        sequence_length = sequence_length.data
    if max_len is None:
        max_len = sequence_length.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return (seq_range_expand < seq_length_expand).float()

def precision_recall_f1(confusion_matrix):
    """Given confusion_matrix in numpy, calculate precision, recall and f1 score.
    """
    dim = confusion_matrix.shape[0]
    TP_FP = np.sum(confusion_matrix, 0)
    TP_FN = np.sum(confusion_matrix, 1)
    print(TP_FP)
    print(TP_FN)
    precision, recall = np.zeros(dim), np.zeros(dim)

    for i in range(dim):
        precision[i] = (confusion_matrix[i, i] / TP_FP[i])
        recall[i] = (confusion_matrix[i, i] / TP_FN[i])
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score
