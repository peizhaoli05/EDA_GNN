# -*- coding: utf-8 -*-
# @File    : final.py
# @Author  : Peizhao Li
# @Contact : peizhaoli05gmail.com 
# @Date    : 2018/11/10

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torchvision import models


class ANet(nn.Module):

    def __init__(self):
        super(ANet, self).__init__()

        self.ANet = models.resnet18(pretrained=True)
        self.ANet.fc = nn.Linear(512, 256)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, pre_crop, cur_crop):
        pre_crop = self.ANet(pre_crop)
        cur_crop = self.ANet(cur_crop)

        crop = torch.tan(F.cosine_similarity(pre_crop, cur_crop))

        pre_feature = pre_crop
        cur_feature = cur_crop

        return crop, pre_feature, cur_feature


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

    def adj_norm(self, adj):
        adj_norm = adj
        adj_t_norm = adj.t()

        return adj_norm, adj_t_norm

    def forward(self, pre, cur, adj):
        pre_ = torch.mm(pre, self.weight)
        cur_ = torch.mm(cur, self.weight)

        adj_norm, adj_t_norm = self.adj_norm(adj)

        pre = torch.mm(adj_norm, cur_)
        cur = torch.mm(adj_t_norm, pre_)

        return pre, cur

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, planes):
        super(GCN, self).__init__()

        self.gc = GraphConvolution(planes, planes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def edge_update(self, pre, cur, adj):
        score = torch.zeros(pre.size(0) * cur.size(0)).cuda()
        adj_ = torch.zeros_like(adj).cuda()

        for i in range(pre.size(0)):
            pre_ = pre[i].unsqueeze(dim=0)
            for j in range(cur.size(0)):
                cur_ = cur[j].unsqueeze(dim=0)
                score_ = torch.tan(F.cosine_similarity(pre_, cur_))
                score[i * cur.size(0) + j] = score_
                adj_[i, j] = score_

        return score, adj_

    def forward(self, pre, cur, adj):
        pre, cur = self.gc(pre, cur, adj)
        score, adj = self.edge_update(pre, cur, adj)

        return score, adj


class final(nn.Module):

    def __init__(self):
        super(final, self).__init__()
        self.embnet = ANet()
        self.gc = GCN(planes=256)

    def forward(self, pre_crop, cur_crop):
        cur_num = len(cur_crop)
        pre_num = len(pre_crop)

        adj1 = torch.zeros(pre_num, cur_num).cuda()
        pre_feature = torch.zeros(pre_num, 256).cuda()
        cur_feature = torch.zeros(cur_num, 256).cuda()
        s0 = torch.zeros(pre_num * cur_num).cuda()

        for i in range(pre_num):
            pre_crop_ = pre_crop[i].cuda().unsqueeze(dim=0)
            for j in range(cur_num):
                cur_crop_ = cur_crop[j].cuda().unsqueeze(dim=0)

                score0_, pre, cur = self.embnet(pre_crop_, cur_crop_)
                adj1[i, j] = score0_
                s0[i * cur_num + j] = score0_
                pre_feature[i, :] = pre
                cur_feature[j, :] = cur

        s3, adj = self.gc(pre_feature, cur_feature, adj1)

        return s0, s3, adj
