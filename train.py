# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:30:03 2020

@author: localuser
"""

import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.nn as nn

STS_CNT = 14


class AccMetric():
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum1 = 0
        self._sum2 = 0
        self._count = 0

    def update(self, targets, o1, o2):
        pred1 = o1.argmax(1)
        pred2 = o2.argmax(1)
        self._sum1 += (pred1 == targets[:, 0]).sum()
        self._sum2 += (pred2 == targets[:, 1]).sum()
        self._count += targets.shape[0]

    def gets(self):
        return self._sum1 / self._count, self._sum2/self._count


class AccMetricSts():
    def __init__(self):
        self.reset()

    def reset(self):
        self._sts0_sum1 = 0
        self._sts1_sum1 = 0
        self._sts2_sum1 = 0
        self._sts3_sum1 = 0
        self._stss_sum = [self._sts0_sum1, self._sts1_sum1,
                          self._sts2_sum1, self._sts3_sum1]

        self._count0 = 0
        self._count1 = 0
        self._count2 = 0
        self._count3 = 0
        self._counts = [self._count0, self._count1,
                        self._count2, self._count3]

    def update(self, targets, o1, o2):
        pred1 = o1.argmax(1)
        pred2 = o2.argmax(1)
        target0 = []
        target1 = []
        target2 = []
        target3 = []
        targetsts = [target0, target1, target2, target3]

        pred1 = o1.argmax(1)
        pred2 = o2.argmax(1)

        for i in range(len(targets)):
            sts = targets[i, 1]
            self._counts[sts] += 1
            targetsts[sts].append(i)
            self._stss_sum[sts] += (pred2[i] == sts)

    def gets(self):
        ret = []
        for i in range(4):
            ret.append(self._stss_sum[i] / self._counts[i])
        return ret


def gen_seq_data(data, sam_len, sts_cnt=STS_CNT):
    length = data.shape[0] - sam_len + 1
    sample_data = []
    for i in range(sam_len):
        sample_data.append(data[i:i+length])

    sample_data = np.hstack(sample_data)
    sample_data = sample_data.reshape((-1, sam_len, sts_cnt))
    sample_data = np.swapaxes(sample_data, 1, 2)
    return sample_data


def train(model, optim, train_loader):
    model.train()
    acc = AccMetric()

    criterion = nn.NLLLoss()

    for data, labels in train_loader:
        X = Variable(data.cuda())
        y1 = Variable(labels[:, 0].cuda())
        y2 = Variable(labels[:, 1].cuda())
        o1, o2 = model(X)

        loss1 = criterion(nn.LogSoftmax()(o1), y1.long())
        loss2 = criterion(nn.LogSoftmax()(o2), y2.long())

        total_loss = loss1+loss2

        acc.update(labels.numpy(), o1.data.cpu().numpy(),
                   o2.data.cpu().numpy())

        optim.zero_grad()
        total_loss.backward()
        optim.step()

    return acc.gets()


def validate(model, testloader):
    model.eval()
    acc = AccMetric()
    for data, labels in testloader:
        X = Variable(data.cuda())
        y1 = Variable(labels[:, 0].cuda())
        y2 = Variable(labels[:, 1].cuda())
        o1, o2 = model(X)
        acc.update(labels.numpy(), o1.data.cpu().numpy(),
                   o2.data.cpu().numpy())

    return acc.gets()


def validate_sts(model, testloader):
    model.eval()
    acc = AccMetricSts()
    for data, labels in testloader:
        X = Variable(data.cuda())
        y1 = Variable(labels[:, 0].cuda())
        y2 = Variable(labels[:, 1].cuda())
        o1, o2 = model(X)
        acc.update(labels.numpy(), o1.data.cpu().numpy(),
                   o2.data.cpu().numpy())

    return acc.gets()
