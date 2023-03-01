import torch
import torch.nn.functional as F


# Loss functions
def loss_co_teaching(y1, y2, y3, target, pick_ratio):
    loss1_all = F.cross_entropy(y1 + y3, target, reduce=False)
    idx1_sorted = torch.argsort(loss1_all.data).cuda()
    loss2_all = F.cross_entropy(y2 + y3, target, reduce=False)
    idx2_sorted = torch.argsort(loss2_all.data).cuda()
    num_remember = int(pick_ratio * len(target))
    idx1_update = idx1_sorted[:num_remember]
    idx2_update = idx2_sorted[:num_remember]
    loss1_update = F.cross_entropy(y1[idx2_update], target[idx2_update])
    loss2_update = F.cross_entropy(y2[idx1_update], target[idx1_update])
    loss1_some = torch.sum(loss1_update) / num_remember
    loss2_some = torch.sum(loss2_update) / num_remember
    return loss1_some, loss2_some


