# -*- coding: utf-8 -*-
# @File    : GCN.py
# @Author  : Peizhao Li
# @Contact : lipeizhao1997@gmail.com 
# @Date    : 2018/10/22

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, pre, cur, adj):
        pre_ = torch.mm(pre, self.weight)
        cur_ = torch.mm(cur, self.weight)

        pre = torch.mm(adj, cur_)
        cur = torch.mm(adj.t(), pre_)

        pre = F.leaky_relu(pre)
        cur = F.leaky_relu(cur)

        return pre, cur

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, planes):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(planes, planes)
        self.gc2 = GraphConvolution(planes, planes)

    def cos_sim(self, pre, cur, adj):
        adj_ = torch.zeros_like(adj).cuda()
        for i in range(pre.size(0)):
            for j in range(cur.size(0)):
                adj_[i, j] = F.cosine_similarity(pre[i:i + 1], cur[j:j + 1])

        return adj_

    def forward(self, pre, cur, adj):
        pre, cur = self.gc1(pre, cur, adj)
        adj = self.cos_sim(pre, cur, adj)
        pre, cur = self.gc2(pre, cur, adj)
        adj = self.cos_sim(pre, cur, adj)

        return adj


class fuckupGCN(nn.Module):

    def __init__(self, planes):
        super(fuckupGCN, self).__init__()

        self.gc1 = GraphConvolution(planes, planes)
        self.gc2 = GraphConvolution(planes, planes)

        self.fc1 = nn.Sequential(nn.Linear(144, 72), nn.LeakyReLU(), nn.Linear(72, 36), nn.LeakyReLU(),
                                 nn.Linear(36, 2))
        self.fc2 = nn.Sequential(nn.Linear(144, 72), nn.LeakyReLU(), nn.Linear(72, 36), nn.LeakyReLU(),
                                 nn.Linear(36, 2))

    def MLP(self, fc, pre, cur, adj):
        score = torch.zeros(pre.size(0) * cur.size(0), 2).cuda()
        adj_ = torch.zeros_like(adj).cuda()

        for i in range(pre.size(0)):
            pre_ = pre[i].unsqueeze_(dim=0)
            for j in range(cur.size(0)):
                cur_ = cur[j].unsqueeze_(dim=0)
                tmp = pre_ - cur_
                score_ = fc(tmp)
                score_ = score_.squeeze_()
                score[i * cur.size(0) + j] = score_
                adj_[i, j] = score_[1]

        return score, adj_

    def forward(self, pre, cur, adj):
        pre, cur = self.gc1(pre, cur, adj)
        score1, adj = self.MLP(self.fc1, pre, cur, adj)
        pre, cur = self.gc2(pre, cur, adj)
        score2, adj = self.MLP(self.fc2, pre, cur, adj)

        return score1, score2, adj
