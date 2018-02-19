from __future__ import print_function, division
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torchnet.meter import ConfusionMeter

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
    var = Variable(var, volatile=volatile)
    if use_cuda:
        var = var.cuda()
    return var

def cuda_tensor_wrapper(tensor):
    if use_cuda:
        tensor = tensor.cuda()
    return tensor

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

def parameter_summary(writer, name, param, global_step):
    if isinstance(param, Variable):
        param = param.data
    writer.add_scalars('Train/' + name, 
        {name + '_mean': torch.mean(param),
        name + '_std': torch.std(param),
        name + '_max': torch.max(param),
        name + '_min': torch.min(param)}, global_step)
    writer.add_histogram(name, param.clone().cpu().numpy(), global_step, bins='doane')

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

def precision_recall_f1(confusion_matrix, average=True):
    """Given confusion_matrix in numpy, calculate precision, recall and f1 score.
    """
    confusion_matrix = confusion_matrix.value()
    dim = confusion_matrix.shape[0]
    TP_FP = np.sum(confusion_matrix, 0)
    TP_FN = np.sum(confusion_matrix, 1)
    precision, recall, f1_score = np.zeros(dim), np.zeros(dim), np.zeros(dim)

    for i in range(dim):
        if confusion_matrix[i, i] == 0:
            precision[i] = 0
            recall[i] = 0
            f1_score[i] = 0
        else:
            precision[i] = (confusion_matrix[i, i] / TP_FP[i])
            recall[i] = (confusion_matrix[i, i] / TP_FN[i])
            f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    if average:
        return np.mean(precision), np.mean(recall), np.mean(f1_score)
    else:
        return precision, recall, f1_score

def get_batch_accuracy(labels, output, lengths):
    correct = 0.0
    total = 0.0
    if len(output.size()) > 2:
        _, output = torch.max(F.softmax(output, 2), 2)
    if isinstance(output, Variable):
        output = output.data
    correct_matrix = torch.eq(output, labels.data).float()
    mask = sequence_mask(lengths)
    correct = torch.sum(correct_matrix * mask)
    total = torch.sum(lengths.data)
    return correct, total

def evaluate(model, dataloader, crf, criterion, f1=False):
    model.eval()
    if f1:
        confusion_matrix = ConfusionMeter(8)
        confusion_matrix.reset()
    correct, total = 0, 0
    loss_total = 0
    for i, (features, labels, _, lengths) in enumerate(dataloader):
        max_length = max(lengths)
        features = cuda_var_wrapper(features[:, :, :max_length], volatile=True)
        labels = cuda_var_wrapper(labels[:, :max_length], volatile=True)
        lengths = cuda_var_wrapper(lengths)
        if crf:
            _, output = model(features, lengths)
            loss = model.forward_alg(features, labels, lengths)
            preds = output
        else:
            output = model(features)
            loss = criterion(output.contiguous().view(-1, 8), labels.contiguous().view(-1))
            _, preds = torch.max(F.softmax(output, 2), 2)
        correct_batch, total_batch = get_batch_accuracy(labels, output, lengths)
        loss_total += loss.data[0] * features.size()[0]
        correct += correct_batch
        total += total_batch
        if f1:
            if isinstance(preds, Variable):
                preds = preds.data
            for pred, truth, length in zip(preds, labels.data, lengths.data):
                confusion_matrix.add(pred[:length], truth[:length])
    if f1:
        # print(confusion_matrix.value())
        precision, recall, f1 = precision_recall_f1(confusion_matrix)
        return correct / total, loss_total, precision, recall, f1
    return correct / total, loss_total

def eval(model, features, labels, lengths, crf, criterion):
    model.eval()
    if crf:
        _, output = model(features, lengths)
        loss = model.forward_alg(features, labels, lengths)
    else:
        output = model(features)
        loss = criterion(output.contiguous().view(-1, 8), labels.contiguous().view(-1))
    loss_batch = loss.data[0] * features.size()[0]
    correct_batch, total_batch = get_batch_accuracy(labels, output, lengths)
    return correct_batch, total_batch, loss_batch

def crf_param(model):
    """Given the model with CRF as the last layer, return the crf transition
    parameters without the START and STOP column and row.
    """
    crf_param = list(model.children())[-1].transitions.cpu().data.numpy()
    crf_param_2 = np.delete(crf_param, 8, 0)
    crf_param_2 = np.delete(crf_param_2, 9, 1)
    return torch.from_numpy(crf_param_2)