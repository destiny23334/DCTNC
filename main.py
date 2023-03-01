import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import cal_noisy_ratio, Iris, Diabetes, HeartStatlog, \
    Ionosphere, Waveform, Biodeg, Segment, Sonar, Spambase, Vehicle, BalanceScale, MyDataset, LabelMe
from loss import loss_co_teaching, loss_co_teaching_orig
from model import DNN512

# which graphics card, 0 or 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0002)  # 学习率
parser.add_argument('--num_iter_per_epoch', type=int, default=400)  # 每个epoch迭代次数
parser.add_argument('--n_epoch', type=int, default=100)  # 进行几轮epoch
parser.add_argument('--alpha', type=float, default=0.4)  # 划分干净集噪声集阈值
parser.add_argument('--exponent', type=float, default=1)  # 超参数
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for '
                         'R(T) in Co-teaching paper.')
parser.add_argument('--epoch_decay_start', type=int, default=80)
args = parser.parse_args()

batch_size = 8
learning_rate = args.learning_rate
input_channel = 3
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


def train_co_teaching(train_loader, epoch, model1, optimizer1, model2, optimizer2, cnn):
    """train two networks through co-teaching in an epoch
    -----
    Args:
        train_loader (DataLoader): 训练集的DataLoader
        epoch (int): 指示当前是哪个epoch
        model1 (Model): Net1
        optimizer1 (): Net1的优化器
        model2 (Model): Net2
        optimizer2 (): Net2的优化器
    --------
    Returns:
        train_acc1 ():
        train_acc2 ():
        ratio_list1 ():
        ratio_list2 ():
        :param cnn:
    """
    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0
    losses1 = 0
    losses2 = 0

    # train networks
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
        out3 = cnn(images)

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


def train_co_teaching_orig(train_loader, epoch, model1, optimizer1, model2, optimizer2):
    """训练两个网络
        emm
    -----
    Args:
        train_loader (DataLoader): 训练集的DataLoader
        epoch (int): 指示当前是哪个epoch
        model1 (Model): Net1
        optimizer1 (): Net1的优化器
        model2 (Model): Net2
        optimizer2 (): Net2的优化器
    --------
    Returns:
        train_acc1 ():
        train_acc2 ():
        ratio_list1 ():
        ratio_list2 ():
    """
    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0
    losses1 = 0
    losses2 = 0

    # 迭代训练两个网络
    for i, (images, labels) in enumerate(train_loader):
        if i > args.num_iter_per_epoch:
            break
        images = Variable(images).cuda().float()
        labels = Variable(labels.long()).cuda().long()

        # Net1的预测
        out1 = model1(images)
        predict1, _ = accuracy(out1, labels, topk=(1, 1))
        correct1 += predict1
        total1 += 1

        # Net2的预测
        out2 = model2(images)
        predict2, _ = accuracy(out2, labels, topk=(1, 1))
        correct2 += predict2
        total2 += 1

        # 计算损失
        loss1, loss2 = \
            loss_co_teaching_orig(out1, out2, labels, 1 - rate_schedule[epoch])

        # 优化
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        # 输出结果
        if (i + 1) % 5 == 0:
            print(
                'Epoch [%d/%d], Co-Teaching, Iter [%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, '
                'Loss1: %.4f, Loss2: %.4f'
                % (epoch + 1, args.n_epoch, i + 1, correct1 / total1, correct2 / total2, loss1.data,
                   loss2.data))

    # Net1和Net2的训练误差
    train_acc1 = float(correct1) / float(total1)
    train_acc2 = float(correct2) / float(total2)
    return train_acc1, train_acc2


def train_normal(clean_loader, cnn, optimizer):
    train_loss = 0.
    train_acc = 0.
    length = 0
    for batch_x, batch_y, idx in clean_loader:
        batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
        out = cnn(batch_x)
        loss = nn.CrossEntropyLoss()(out, batch_y)
        train_loss += loss.data
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data
        length += len(batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = train_acc / length
    return train_acc


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


def evaluate(test_loader, models):
    acc = []
    losses = []
    for model in models:
        model.eval()  # Change model to 'eval' mode.
        eval_loss = 0.
        eval_acc = 0.
        eval_length = 0
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = Variable(batch_x, volatile=True).cuda().float(), \
                Variable(batch_y, volatile=True).cuda().long()
            out = model(batch_x)
            loss = nn.CrossEntropyLoss()(out, batch_y)
            eval_loss += loss.data
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.data
            eval_length += len(batch_y)
        acc.append(100 * eval_acc / eval_length)
        losses.append(eval_loss)
    return acc, losses


def evaluate_tri(test_loader, models):
    eval_loss = 0.
    eval_acc = 0.
    eval_length = 0
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True).cuda().float(), \
            Variable(batch_y, volatile=True).cuda().long()
        out = 0
        for model in models:
            model.eval()  # Change model to 'eval' mode.
            out = model(batch_x) + out

        out = out / 3
        loss = nn.CrossEntropyLoss()(out, batch_y)
        eval_loss += loss.data
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data
        eval_length += len(batch_y)
    return (100 * eval_acc / eval_length), eval_loss


def split_clean_noisy(dataset: MyDataset, alpha):
    """ 根据gini划分干净集和噪声集

    Parameters
    ----------
    dataset:
    alpha: 干净集与噪声集的比例

    Returns
    -------
    clean_idx: 干净集样本的索引
    noise_idx: 噪声集样本的索引
    """

    gini = dataset.gini  # 获取所有样本的gini
    sorted_idx = np.argsort(gini)  # 对gini排序，越小约纯

    integrate_labels = np.array(dataset.integrate_labels, dtype=int)
    labels, counts = np.unique(integrate_labels, return_counts=True)
    num_clean = (counts * alpha).astype(int)
    select_clean = np.zeros([len(labels)], dtype=int)
    clean_idx = []
    noisy_idx = []

    for i in range(len(sorted_idx)):  # 按比例选择
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
    train_data_loader = DataLoader(train_datasets, batch_size=batch_size, drop_last=True)
    # test_datasets = data_type(isTruth=True)
    # valid_data_loader = DataLoader(test_datasets, batch_size=batch_size)

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
        print('epoch {}'.format(epoch + 1))
        train_loss = 0.
        train_acc = 0.
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

        print("Epoch [%d/%d], Clean-Training, Train acc1: %.6f, Train loss: %.6f"
              % (epoch + 1, epochs, train_acc / (len(clean_train_dataset)),
                 train_loss / (len(clean_train_dataset))))

    # train f_noise1 and f_noise2
    for epoch in range(epochs):
        train_acc1, train_acc2, loss1, loss2 = \
            train_co_teaching(noisy_train_loader, epoch, dnn1, opti1, dnn2, opti2, dnn_clean)
        print("Epoch [%d/%d], Co-Teaching-Train, Train acc1: %.6f, "
              "Train acc2: %.6f, Loss1: %.6f, Loss2: %.6f"
              % (epoch + 1, epochs, train_acc1, train_acc2, loss1, loss2))

    # +----------------------------------------+
    # |              correction                |
    # +----------------------------------------+
    corrected_labels = []
    truth = noisy_train_dataset.true_labels  # ground truth of noise set
    aggregation_label = noisy_train_dataset.integrate_label
    count = 0
    for instance, label in noisy_correct_loader:
        instance = Variable(instance).float().cuda()
        labels = Variable(labels).long().cuda()

        predict_clean = get_predict(dnn_clean(instance))
        predict_noise1 = get_predict(dnn1(instance))
        predict_noise2 = get_predict(dnn2(instance))

        if predict_clean.data == predict_noise1.data and predict_clean.data == predict_noise2.data:
            corrected_labels.append(predict_clean.cpu().data.item())
        else:
            corrected_labels.append(aggregation_label[count])
        count += 1

    f.write(str() + '\n')
    print("Correct" % ())


if __name__ == '__main__':
    dataset = [BalanceScale, Biodeg, Diabetes, HeartStatlog, Ionosphere, Iris,
               Segment, Sonar, Spambase, Vehicle, Waveform, LabelMe]
    f = open('./result.txt', 'w')
    for i_dataset in dataset:
        run(i_dataset)
    f.close()
