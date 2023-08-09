# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 23:20
# @Author  : Curry
# @File    : loss.py

import torch
import torch.nn.functional as F
import torch.nn as nn

class smooth_CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''

    def __init__(self, label_smooth=None, class_num=2):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12

        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # 转换成one-hot

            # label smoothing
            # 实现 1
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num
            # 实现 2
            # implement 2
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)

        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))

        return loss.mean()


class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=None, reduction='elementwise_mean', w0=1, w1=15):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.w0 = w0
        self.w1 = w1

    def forward(self, _input, target):
        pt = F.softmax(_input, dim=-1)
        target = target.unsqueeze(-1)
        loss = - (1 - pt) ** self.gamma * target * self.w1 * torch.log(pt) - \
            pt ** self.gamma * (1 - target) * self.w0 * torch.log(1 - pt)
        if self.alpha:
            loss = loss * self.alpha
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def focal_lossv1(logits, labels, gamma=2):
    r"""
    focal loss for multi classification（第一版）
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    """

    # pt = F.softmax(logits, dim=-1)  # 直接调用可能会溢出
    # 一个不会溢出的 trick
    log_pt = F.log_softmax(logits, dim=-1)  # 这里相当于 CE loss
    pt = torch.exp(log_pt)  # 通过 softmax 函数后打的分
    labels = labels.view(-1, 1)  # 多加一个维度，为使用 gather 函数做准备
    pt = pt.gather(1, labels)  # 挑选出真实值对应的 softmax 打分，也可以使用独热编码实现
    ce_loss = -torch.log(pt)
    weights = (1 - pt) ** gamma
    fl = weights * ce_loss
    fl = fl.mean()
    return fl

def build_loss():
    # CELoss = torch.nn.CrossEntropyLoss()
    CELoss = smooth_CELoss(label_smooth=0.05, class_num=2)
    bceFocalLoss = BCEFocalLoss(gamma=2, alpha=0.25)

    return {"CELoss":CELoss, "bceFocalLoss": bceFocalLoss}
