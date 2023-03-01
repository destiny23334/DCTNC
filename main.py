import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import Iris, Diabetes, HeartStatlog, Ionosphere, Waveform, \
    Biodeg, Segment, Sonar, Spambase, Vehicle, BalanceScale, MyDataset, LabelMe
from loss import loss_co_teaching
from model import DNN512

# which graphics card, 0 or 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--alpha', type=float, default=0.4)
parser.add_argument('--exponent', type=float, default=1)
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for '
                         'R(T) in Co-teaching paper.')
parser.add_argument('--epoch_decay_start', type=int, default=80)
args = parser.parse_args()

batch_size = 8
learning_rate = args.learning_rate
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
alpha = args.alpha
forget_rate = 0.3
rate_schedule = np.ones(args.n_epoch) * forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

# Adjust learning rate and betas for Adam Optimizer
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)  # Only change beta1


def get_predict(out):
    output = F.softmax(out, dim=1)
    _, predict = output.topk(1, 1, True, True)
    return predict.t()


def train_co_teaching(train_loader, epoch, model1, optimizer1, model2, optimizer2, dnn):
    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0
    losses1 = 0
    losses2 = 0

    # train networks
    model1.train()
    model2.train()
    dnn.eval()
    for i, (images, labels) in enumerate(train_loader):
        if i > args.num_iter_per_epoch:
            break
        images = Variable(images).cuda().float()
        labels = Variable(labels.long()).cuda().long()

        # predicts of f_noise1
        out1 = model1(images)
        predict1, _ = accuracy(out1, labels, topk=(1, 1))
        correct1 += predict1
        total1 += 1

        # predicts of f_noise2
        out2 = model2(images)
        predict2, _ = accuracy(out2, labels, topk=(1, 1))
        correct2 += predict2
        total2 += 1

        # predicts of f_clean
        out3 = dnn(images)

        # calculate loss
        loss1, loss2 = \
            loss_co_teaching(out1, out2, out3, labels, 1 - rate_schedule[epoch])
        losses1 += loss1.data
        losses2 += loss2.data

        # optimize
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        # if (i + 1) % 5 == 0:
        #     print('Epoch [%d/%d], Co-Teaching, Iter [%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, '
        #         'Loss1: %.4f, Loss2: %.4f'
        #         % (epoch + 1, args.n_epoch, i + 1, correct1 / total1, correct2 / total2, loss1.data,
        #            loss2.data))

    train_acc1 = float(correct1) / float(total1)
    train_acc2 = float(correct2) / float(total2)
    loss1 = losses1 / total1
    loss2 = losses2 / total2
    return train_acc1, train_acc2, loss1, loss2


def accuracy(out, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # output = F.softmax(out, dim=1)
    output = out
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def split_clean_noisy(dataset: MyDataset, alpha: float):
    """ obtain noise set and clean set according to gini"""
    gini = dataset.gini
    sorted_idx = np.argsort(gini)

    integrate_labels = np.array(dataset.integrate_labels, dtype=int)
    labels, counts = np.unique(integrate_labels, return_counts=True)
    num_clean = (counts * alpha).astype(int)
    select_clean = np.zeros([len(labels)], dtype=int)
    clean_idx = []
    noisy_idx = []

    for i in range(len(sorted_idx)):
        idx = sorted_idx[i]
        label = integrate_labels[idx]
        if select_clean[label] < num_clean[label]:
            select_clean[label] += 1
            clean_idx.append(idx)
        else:
            noisy_idx.append(idx)
    return clean_idx, noisy_idx


def run(idx_dataset: MyDataset):
    # +----------------------------------------+
    # |          data preprocessing            |
    # +----------------------------------------+
    f.write(idx_dataset.__name__ + '\t')
    data_type = idx_dataset
    train_datasets = data_type()
    valid_datasets = data_type(isTruth=True)
    valid_data_loader = DataLoader(valid_datasets, batch_size=batch_size)

    # noise filter
    clean_idx, noisy_idx = split_clean_noisy(train_datasets, alpha)

    # reload clean set and noise set
    clean_train_dataset = data_type(idx=clean_idx)
    noisy_train_dataset = data_type(idx=noisy_idx)
    noisy_train_loader = DataLoader(dataset=noisy_train_dataset, batch_size=batch_size,
                                    drop_last=True, shuffle=True)
    clean_train_loader = DataLoader(dataset=clean_train_dataset, batch_size=batch_size,
                                    drop_last=True, shuffle=True)
    noisy_correct_loader = DataLoader(dataset=noisy_train_dataset, batch_size=1,
                                      drop_last=False, shuffle=False)
    clean_set = data_type(idx=clean_idx)

    # +----------------------------------------+
    # |              init model                |
    # +----------------------------------------+
    epochs = args.n_epoch
    loss_func = nn.CrossEntropyLoss()
    dnn_clean = DNN512(clean_train_dataset.num_feature, clean_train_dataset.num_classes)
    dnn_clean.cuda()
    optimizer = optim.Adam(dnn_clean.parameters(), lr=learning_rate)

    # f_noise1 and f_noise2
    dnn1 = DNN512(clean_train_dataset.num_feature, clean_train_dataset.num_classes)
    dnn1.cuda()
    opti1 = optim.Adam(dnn1.parameters(), lr=learning_rate)
    dnn2 = DNN512(clean_train_dataset.num_feature, clean_train_dataset.num_classes)
    dnn2.cuda()
    opti2 = optim.Adam(dnn2.parameters(), lr=learning_rate)

    # +----------------------------------------+
    # |              build model               |
    # +----------------------------------------+

    # train f_clean
    for epoch in range(epochs):
        dnn_clean.train()
        train_loss = 0.
        train_acc = 0.
        valid_loss = 0.
        valid_acc = 0.
        for batch_x, batch_y in clean_train_loader:
            batch_x, batch_y = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
            out = dnn_clean(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.data
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        dnn_clean.eval()
        for batch_x, batch_y in valid_data_loader:
            batch_x, batch_y = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
            out = dnn_clean(batch_x)
            loss = loss_func(out, batch_y)
            valid_loss += loss.data
            pred = torch.max(out, 1)[1]
            valid_correct = (pred == batch_y).sum()
            valid_acc += valid_correct.data
        print("Epoch [%d/%d], Clean-Training, Train acc: %.6f, "
              "Train loss: %.6f, Valid acc: %.6f, Valid loss: %.6f"
              % (epoch + 1, epochs, train_acc / len(clean_train_dataset),
                 train_loss / len(clean_train_dataset), valid_acc / len(valid_datasets),
                 valid_loss / len(valid_datasets)))

    # train f_noise1 and f_noise2
    for epoch in range(epochs):
        # train
        train_acc1, train_acc2, loss1, loss2 = \
            train_co_teaching(noisy_train_loader, epoch, dnn1, opti1, dnn2, opti2, dnn_clean)
        # evaluate
        valid_acc1 = 0.
        valid_acc2 = 0.
        dnn1.eval()
        dnn2.eval()
        for batch_x, batch_y in valid_data_loader:
            batch_x, batch_y = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
            out1 = dnn1(batch_x)
            out2 = dnn2(batch_x)
            pred1 = torch.max(out1, 1)[1]
            pred2 = torch.max(out2, 1)[1]
            valid_correct1 = (pred1 == batch_y).sum()
            valid_correct2 = (pred2 == batch_y).sum()
            valid_acc1 += valid_correct1.data
            valid_acc2 += valid_correct2.data
        print("Epoch [%d/%d], Co-Teaching-Train, Train acc1: %.6f, "
              "Train acc2: %.6f, Valid acc1: %.6f, Valid acc2: %.6f"
              % (epoch + 1, epochs, train_acc1, train_acc2,
                 valid_acc1 / len(valid_datasets), valid_acc2 / len(valid_datasets)))

    # +----------------------------------------+
    # |              correction                |
    # +----------------------------------------+
    corrected_labels = []
    truth = noisy_train_dataset.true_labels  # ground truth of noise set
    count = 0
    for instance, label in noisy_correct_loader:
        instance = Variable(instance).float().cuda()
        label = Variable(label).long().cuda()

        predict_clean = get_predict(dnn_clean(instance))
        predict_noise1 = get_predict(dnn1(instance))
        predict_noise2 = get_predict(dnn2(instance))

        if predict_clean.data == predict_noise1.data and predict_clean.data == predict_noise2.data:
            corrected_labels.append(predict_clean.cpu().data.item())
        else:
            corrected_labels.append(label.cpu().data.item())
        count += 1

    error = 0
    for i in range(len(noisy_train_dataset)):
        if truth[i] != corrected_labels[i]:
            error += 1
    for i in range(len(clean_set)):
        if clean_set.integrate_labels[i] != clean_set.true_labels[i]:
            error += 1
    f.write(str(1 - error / (len(noisy_train_dataset) + len(clean_set))) + '\n')


if __name__ == '__main__':
    dataset = [BalanceScale, Biodeg, Diabetes, HeartStatlog, Ionosphere, Iris,
               Segment, Sonar, Spambase, Vehicle, Waveform, LabelMe]
    f = open('./result.txt', 'w')
    for i_dataset in dataset:
        run(i_dataset)
    f.close()
