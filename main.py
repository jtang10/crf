from __future__ import print_function, division
import time
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
from data_loading import Protein_Dataset
from mufold import MUFold_ss
from resnet import Inception_ResNet
from model import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch implementation of Mufold-ss paper')
parser.add_argument('run', metavar='DIR', help='directory to save the summary and model')
parser.add_argument('model', choices=['mufold', 'resnet'], help='choose model between mufold and densenet')
parser.add_argument('optimizer', choices=['sgd', 'adam'], default='adam', help='choose optimizer')
parser.add_argument('init', choices=['uniform', 'normal', 'none'])
parser.add_argument('-l', '--local', action='store_false', default=True, help='if specified, use local data root')
parser.add_argument('--n_cnn', default=4, type=int, help='number of resnet layers')
parser.add_argument('--crf', action='store_true', default=False, help="If specified, CRF will be added after nn")
parser.add_argument('-e', '--epochs', default=15, type=int, metavar='N', help='number of epochs to run (default: 15)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='batch size of training data (default: 64)')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate (default: 1e-3)')
parser.add_argument('--lr_decay', default=0.05, type=float, help='the learning rate decay ratio (default: 0.05)')
parser.add_argument('--patience', default=5, type=int, help='patience for early stopping (default: 5)')
parser.add_argument('--dropout', default=0.5, type=float, metavar='N', help='dropout factor. default: 0.4')
parser.add_argument('--adjust_lr', action='store_true', default=False, help="If specified, adjust lr based on validation set accuracy")
parser.add_argument('--least_iter', default=13, type=int, help='guaranteed number of training iterations (default: 30)')
parser.add_argument('--nesterov', action='store_true', default=False, help="If specified, use nesterov in SGD")
parser.add_argument('-c', '--clean', action='store_true', default=False, help="If specified, clear the summary directory first")
parser.add_argument('--reload', action='store_true', default=False, help="If specified, retrain a saved model")
parser.add_argument('-v', '--verbose', action='store_true', default=False, help="If specified, print some information")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using GPU")
else:
    print("Using CPU")

# Initialize dataloader
if args.local:
    data_root = '/share/data/jinbodata/jtang7/ProteinProperty_Project'
else:
    data_root = os.path.expanduser('../data/ProteinProperty_Project')
SetOf7604Proteins_path = data_root + '/SetOf7604Proteins/'
CASP11_path = data_root + '/CASP11/'
CASP12_path = data_root + '/CASP12/'

train_dataset = Protein_Dataset(SetOf7604Proteins_path, 'trainList')
valid_dataset = Protein_Dataset(SetOf7604Proteins_path, 'validList')
test_dataset = Protein_Dataset(SetOf7604Proteins_path, 'testList', padding=True)
CASP11_dataset = Protein_Dataset(CASP11_path, 'proteinList', padding=True)
CASP12_dataset = Protein_Dataset(CASP12_path, 'proteinList', padding=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
CASP11_loader = DataLoader(CASP11_dataset, batch_size=64, shuffle=False, num_workers=4)
CASP12_loader = DataLoader(CASP12_dataset, batch_size=64, shuffle=False, num_workers=4)


# Save model information
model_name = 'mufold' if args.model == 'mufold' else 'resnet'
if args.crf:
    model_name += '_crf'
if args.reload:
    model_name += '_reload'
save_model_dir = os.path.join(os.getcwd(), "saved_model", model_name)
writer_path = os.path.join("logger", model_name, args.run)
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)
model_name = [model_name, args.run, args.init, 'epochs', str(args.epochs),
              args.optimizer, 'lr', str(args.lr), 'b', str(args.batch_size),
              'dropout', str(args.dropout)]
model_name = '_'.join(model_name)
model_path = os.path.join(save_model_dir, model_name)
if args.clean:
    if os.path.exists(writer_path):
        shutil.rmtree(writer_path, ignore_errors=True)
    if os.path.exists(model_path):
        os.remove(model_path)
print('writer_path:', writer_path)
print('save_model_dir:', save_model_dir)
print('model_name:', model_name)


if args.crf:
    model = CNN_CRF(args.dropout, args.model, args.n_cnn)
else:
    model = Inception_ResNet(n_cnn=args.n_cnn) if args.model == 'resnet' else MUFold_ss(dropout=args.dropout)

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=args.nesterov)

criterion = nn.CrossEntropyLoss()
lr_lambda = lambda epoch: 1 / (1 + (epoch + 1) * args.lr_decay)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

def init_weight(m):
    if type(m) == nn.Conv1d or type(m) == nn.Linear:
        if args.init == 'normal':
            nn.init.kaiming_normal(m.weight.data)
        if args.init == 'uniform':
            nn.init.kaiming_uniform(m.weight.data)

model.apply(init_weight)

if use_cuda:
    model.cuda()
    if not args.crf:
        criterion.cuda()
load_trained_model = save_model_dir + 'resnet_crf_epochs_15_adam_lr_0.001_dropout_0.5_best'
if args.reload:
    model.load_state_dict(torch.load(load_trained_model))

writer = SummaryWriter(log_dir=writer_path)
best_valid_acc = 0
patience = 0
len_train = len(train_dataset)
len_valid = len(valid_dataset)
len_test = len(test_dataset)
start = time.time()

for epoch in range(args.epochs):
    correct_train, total_train = 0, 0
    loss_train = 0
    start_epoch = time.time()
    for i, (features, labels, _, lengths) in enumerate(train_loader):
        model.train()
        t_train0 = time.time() - start_epoch
        max_length = max(lengths)
        features = cuda_var_wrapper(features[:, :, :max_length])
        labels = cuda_var_wrapper(labels[:, :max_length])
        lengths = cuda_var_wrapper(lengths)
        if args.crf:
            cnn_output, output = model(features, lengths)
            loss = model.forward_alg(features, labels, lengths)
        else:
            output = model(features)
            loss = criterion(output.contiguous().view(-1, 8), labels.contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t_train1 = time.time() - start_epoch
        correct_batch, total_batch = get_batch_accuracy(labels, output, lengths)
        loss_train += loss.data[0] * args.batch_size
        correct_train += correct_batch
        total_train += total_batch
        niter = epoch * len_train + i * args.batch_size
        if i % 10 == 0:
            if args.crf:
                parameter_summary(writer, 'cnn_output', cnn_output, niter)
                parameter_summary(writer, 'crf_param', crf_param(model), niter)
            else:
                parameter_summary(writer, 'cnn_output', output, niter)
            if args.verbose:
                print ('Epoch [{}/{}], Batch_time: {:.2f}, Iter [{}/{}], Loss: {:.4f}' \
                   .format(epoch+1, args.epochs, time.time() - start_epoch, i, len_train//args.batch_size, loss.data[0]))

    accuracy_train = correct_train / total_train
    accuracy_valid, loss_valid = evaluate(model, valid_loader, args.crf, criterion)
    loss_train /= len_train
    loss_valid /= len_valid
    epoch_time = time.time() - start_epoch
    writer.add_scalars('data/accuracy_group', {'accuracy_train': accuracy_train,
                                               'accuracy_valid': accuracy_valid}, epoch + 1)
    writer.add_scalars('data/loss_group', {'loss_train': loss_train,
                                           'loss_valid': loss_valid}, epoch + 1)
    print("Epoch {}: time: {:.1f}s/{:.1f}s, Training loss {:.4f}, accuracy {:.4f}; Validation loss {:.4f}, accuracy {:.4f}" \
          .format(epoch + 1, t_train1, epoch_time, loss_train, accuracy_train, loss_valid, accuracy_valid))

    if args.adjust_lr:
        scheduler.step(accuracy_valid)
    if accuracy_valid > best_valid_acc:
        patience = 0
        best_valid_acc = accuracy_valid
        torch.save(model.state_dict(), model_path + '_best')
    else:
        patience += 1
        if patience >= args.patience and epoch + 1 >= args.least_iter:
            print("Early stopping activated")
            break

print("Time spent on training: {:.2f}s".format(time.time() - start))
print("Best validation accuracy: {:.4f}".format(best_valid_acc))
torch.save(model.state_dict(), model_path)
writer.close()

model.load_state_dict(torch.load(model_path + '_best'))
accuracy_test, _, p_test, r_test, f1_test = evaluate(model, test_loader, args.crf, criterion, f1=True)
accuracy_CASP11, _, p_11, r_11, f1_11 = evaluate(model, CASP11_loader, args.crf, criterion, f1=True)
accuracy_CASP12, _, p_12, r_12, f1_12 = evaluate(model, CASP12_loader, args.crf, criterion, f1=True)
print("Model for best validation accuracy:")
print("Test accuracy {:.3f}, precision/recall/f1: {:.3f},{:.3f},{:.3f}".format(accuracy_test, p_test, r_test, f1_test))
print("CASP11 accuracy {:.3f}, precision/recall/f1: {:.3f},{:.3f},{:.3f}".format(accuracy_CASP11, p_11, r_11, f1_11))
print("CASP12 accuracy {:.3f}, precision/recall/f1: {:.3f},{:.3f},{:.3f}".format(accuracy_CASP12, p_12, r_12, f1_12))

model.load_state_dict(torch.load(model_path))
accuracy_test, _, p_test, r_test, f1_test = evaluate(model, test_loader, args.crf, criterion, f1=True)
accuracy_CASP11, _, p_11, r_11, f1_11 = evaluate(model, CASP11_loader, args.crf, criterion, f1=True)
accuracy_CASP12, _, p_12, r_12, f1_12 = evaluate(model, CASP12_loader, args.crf, criterion, f1=True)
print("Model for best validation accuracy:")
print("Test accuracy {:.3f}, precision/recall/f1: {:.3f},{:.3f},{:.3f}".format(accuracy_test, p_test, r_test, f1_test))
print("CASP11 accuracy {:.3f}, precision/recall/f1: {:.3f},{:.3f},{:.3f}".format(accuracy_CASP11, p_11, r_11, f1_11))
print("CASP12 accuracy {:.3f}, precision/recall/f1: {:.3f},{:.3f},{:.3f}".format(accuracy_CASP12, p_12, r_12, f1_12))