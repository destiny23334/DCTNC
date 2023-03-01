import torch
import torch.nn.functional as F
from fcloss import MultiFocalLoss


# Loss functions
def loss_co_teaching(y1, y2, y3, target, pick_ratio):
    """计算co_teaching损失

    -----
    Args:
        y1 (Tensor): shape of (batch_size, num_class)，Net1的输出结果
        y2 (Tensor): shape同y1，Net2的输出结果
        y3 (Tensor): shape同y1，NetClean的输出结果
        target (Tensor): shape of (batch_size)，集成标，target[i] < num_class
        pick_ratio (double): 每次选择小损失样本的个数
        # idx (): 一个batch数据在原始数据集上的索引
        # is_clean (): 该数据是否为噪声，用于输出每轮
    --------
    Returns:
        loss1_some (Tensor): int，Net1在小损失数据子集上的平均交叉熵损失
        loss2_some (Tensor): int，Net2损失同上
        noisy_ratio1 (Tensor): bool，用于训练Net2的那部分数据的噪声比
        noisy_ratio1 (Tensor): bool，用于训练Net1的那部分数据的噪声比
    """
    # 计算每个mini-batch上的损失，并将样本按损失增序排序
    # loss1_all = MultiFocalLoss(size_average=False)(y1+y3, target)
    loss1_all = F.cross_entropy(y1 + y3, target, reduce=False)
    idx1_sorted = torch.argsort(loss1_all.data).cuda()

    # loss2_all = MultiFocalLoss(size_average=False)(y2+y3, target)
    loss2_all = F.cross_entropy(y2 + y3, target, reduce=False)
    idx2_sorted = torch.argsort(loss2_all.data).cuda()

    # 小损失样本的数量
    num_remember = int(pick_ratio * len(target))

    # 小损失样本集的噪声率
    # noisy_ratio_1 = torch.sum(is_clean[idx[idx1_sorted[:num_remember]]]) / float(num_remember)
    # noisy_ratio_2 = torch.sum(is_clean[idx[idx2_sorted[:num_remember]]]) / float(num_remember)

    # 用于训练网络的样本的下标
    idx1_update = idx1_sorted[:num_remember]
    idx2_update = idx2_sorted[:num_remember]

    # 计算训练的损失，用于反向传播
    # loss1_update = MultiFocalLoss()(y1[idx2_update], target[idx2_update])
    # loss2_update = MultiFocalLoss()(y2[idx1_update], target[idx1_update])
    loss1_update = F.cross_entropy(y1[idx2_update], target[idx2_update])
    loss2_update = F.cross_entropy(y2[idx1_update], target[idx1_update])
    loss1_some = torch.sum(loss1_update) / num_remember
    loss2_some = torch.sum(loss2_update) / num_remember

    return loss1_some, loss2_some


def loss_co_teaching_orig(y1, y2, target, pick_ratio):
    """计算co_teaching损失

    -----
    Args:
        y1 (Tensor): shape of (batch_size, num_class)，Net1的输出结果
        y2 (Tensor): shape of (batch_size, num_class)，Net2的输出结果
        target (Tensor): shape of (batch_size)，集成标，target[i] < num_class
        pick_ratio (double): 每次选择小损失样本的个数
        # idx (): 一个batch数据在原始数据集上的索引
        # is_clean (): 该数据是否为噪声，用于输出每轮
    --------
    Returns:
        loss1_some (Tensor): int，Net1在小损失数据子集上的平均交叉熵损失
        loss2_some (Tensor): int，Net2损失同上
        noisy_ratio1 (Tensor): bool，用于训练Net2的那部分数据的噪声比
        noisy_ratio1 (Tensor): bool，用于训练Net1的那部分数据的噪声比
    """
    # 计算每个mini-batch上的损失，并将样本按损失增序排序
    # loss1_all = MultiFocalLoss(size_average=False)(y1+y3, target)
    loss1_all = F.cross_entropy(y1, target, reduce=False)
    idx1_sorted = torch.argsort(loss1_all.data).cuda()

    # loss2_all = MultiFocalLoss(size_average=False)(y2+y3, target)
    loss2_all = F.cross_entropy(y2, target, reduce=False)
    idx2_sorted = torch.argsort(loss2_all.data).cuda()

    # 小损失样本的数量
    num_remember = int(pick_ratio * len(target))

    # 小损失样本集的噪声率
    # noisy_ratio_1 = torch.sum(is_clean[idx[idx1_sorted[:num_remember]]]) / float(num_remember)
    # noisy_ratio_2 = torch.sum(is_clean[idx[idx2_sorted[:num_remember]]]) / float(num_remember)

    # 用于训练网络的样本的下标
    idx1_update = idx1_sorted[:num_remember]
    idx2_update = idx2_sorted[:num_remember]

    # 计算训练的损失，用于反向传播
    # loss1_update = MultiFocalLoss()(y1[idx2_update], target[idx2_update])
    # loss2_update = MultiFocalLoss()(y2[idx1_update], target[idx1_update])
    loss1_update = F.cross_entropy(y1[idx2_update], target[idx2_update])
    loss2_update = F.cross_entropy(y2[idx1_update], target[idx1_update])
    loss1_some = torch.sum(loss1_update) / num_remember
    loss2_some = torch.sum(loss2_update) / num_remember

    return loss1_some, loss2_some


def loss_normal(y, target):
    return F.cross_entropy(y, target)
