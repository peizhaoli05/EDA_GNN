# -*- coding: utf-8 -*-
# @File    : net_1024.py
# @Author  : Peizhao Li
# @Contact : lipeizhao1997@gmail.com 
# @Date    : 2018/10/24

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class convblock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(convblock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        return out


class ANet_2(nn.Module):

    def __init__(self):
        super(ANet_2, self).__init__()
        self.ndf = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = convblock(32, 48)
        self.conv3 = convblock(48, 64)
        self.conv4 = convblock(64, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.max_pool2d(self.conv4(x), 2)

        x = F.avg_pool2d(x, kernel_size=(5, 2))
        x = x.view(x.size(0), -1)

        return x


class ANet(nn.Module):

    def __init__(self):
        super(ANet, self).__init__()
        self.ndf = 32

        self.conv1 = nn.Conv2d(3, self.ndf, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.ndf, int(self.ndf * 1.5), kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(int(self.ndf * 1.5), self.ndf * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.relu(x)
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.relu(x)
        x = F.max_pool2d(self.conv4(x), 2)
        x = F.relu(x)

        x = F.avg_pool2d(x, kernel_size=(5, 2))
        x = x.view(x.size(0), -1)

        return x


class LSTM(nn.Module):

    def __init__(self, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :].clone()

        return x


class embnet(nn.Module):

    def __init__(self):
        super(embnet, self).__init__()
        self.ANet = ANet()
        self.lstm = LSTM(hidden_size=16)
        self.fc = nn.Linear(2, 16, bias=False)

        self.crop_fc1 = nn.Linear(128, 64, bias=False)
        self.crop_fc2 = nn.Linear(64, 32, bias=False)
        self.crop_fc3 = nn.Linear(32, 1, bias=False)

        self.coord_fc1 = nn.Linear(16, 8, bias=False)
        self.coord_fc2 = nn.Linear(8, 1, bias=False)

        self.com = nn.Linear(2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, pre_crop, cur_crop, pre_coord, cur_coord):
        pre_crop = self.ANet(pre_crop)
        cur_crop = self.ANet(cur_crop)
        pre_coord = self.lstm(pre_coord)
        cur_coord = self.fc(cur_coord)

        temp_crop = pre_crop.sub(cur_crop)
        temp_coord = pre_coord.sub(cur_coord)

        crop = F.relu(self.crop_fc1(temp_crop))
        crop = F.relu(self.crop_fc2(crop))
        crop = self.crop_fc3(crop)

        coord = F.relu(self.coord_fc1(temp_coord))
        coord = self.coord_fc2(coord)

        com = torch.cat((crop, coord), dim=1)
        com = self.com(com)

        pre_feature = torch.cat((pre_crop, pre_coord), dim=1)
        cur_feature = torch.cat((cur_crop, cur_coord), dim=1)

        return com, crop, coord, pre_feature, cur_feature


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
        adj_norm = F.softmax(adj, dim=1)
        adj_t_norm = F.softmax(adj.t(), dim=1)

        return adj_norm, adj_t_norm

    def forward(self, pre, cur, adj):
        pre_ = torch.mm(pre, self.weight)
        cur_ = torch.mm(cur, self.weight)

        adj_norm, adj_t_norm = self.adj_norm(adj)

        pre = torch.mm(adj_norm, cur_)
        cur = torch.mm(adj_t_norm, pre_)

        pre = F.relu_(pre)
        cur = F.relu_(cur)

        return pre, cur

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, planes):
        super(GCN, self).__init__()

        self.gc = GraphConvolution(planes, planes)
        # self.gc2 = GraphConvolution(planes, planes)

        self.fc1 = nn.Sequential(nn.Linear(144, 72, bias=False), nn.ReLU(), nn.Linear(72, 36, bias=False), nn.ReLU(),
                                 nn.Linear(36, 1, bias=False))
        # self.fc2 = nn.Sequential(nn.Linear(144, 72, bias=False), nn.ReLU(), nn.Linear(72, 36, bias=False), nn.ReLU(),
        #                          nn.Linear(36, 1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def MLP(self, fc, pre, cur, adj):
        score = torch.zeros(pre.size(0) * cur.size(0)).cuda()
        adj_ = torch.zeros_like(adj).cuda()

        for i in range(pre.size(0)):
            pre_ = pre[i].unsqueeze(dim=0)
            for j in range(cur.size(0)):
                cur_ = cur[j].unsqueeze(dim=0)
                temp = pre_.sub(cur_)
                score_ = fc(temp)
                score_ = score_.squeeze()
                score[i * cur.size(0) + j] = score_
                adj_[i, j] = score_

        return score, adj_

    def forward(self, pre, cur, adj):
        pre, cur = self.gc(pre, cur, adj)
        score, adj = self.MLP(self.fc1, pre, cur, adj)
        # pre, cur = self.gc2(pre, cur, adj)
        # score2, adj = self.MLP(self.fc2, pre, cur, adj)

        return score, adj


class net_1024(nn.Module):

    def __init__(self):
        super(net_1024, self).__init__()
        self.embnet = embnet()
        self.gc = GCN(planes=144)

    def forward(self, pre_crop, cur_crop, pre_motion, cur_motion):
        cur_num = len(cur_crop)
        pre_num = len(pre_crop)

        adj1 = torch.zeros(pre_num, cur_num).cuda()
        pre_feature = torch.zeros(pre_num, 144).cuda()
        cur_feature = torch.zeros(cur_num, 144).cuda()
        s0 = torch.zeros(pre_num * cur_num).cuda()
        s1 = torch.zeros(pre_num * cur_num).cuda()
        s2 = torch.zeros(pre_num * cur_num).cuda()

        for i in range(pre_num):
            pre_crop_ = pre_crop[i].cuda().unsqueeze(dim=0)
            pre_motion_ = pre_motion[i].cuda().unsqueeze(dim=0)
            for j in range(cur_num):
                cur_crop_ = cur_crop[j].cuda().unsqueeze(dim=0)
                cur_motion_ = cur_motion[j].cuda().unsqueeze(dim=0)

                score0_, score1_, score2_, pre, cur = self.embnet(pre_crop_, cur_crop_, pre_motion_, cur_motion_)
                adj1[i, j] = score0_
                s0[i * cur_num + j] = score0_
                s1[i * cur_num + j] = score1_
                s2[i * cur_num + j] = score2_
                pre_feature[i, :] = pre
                cur_feature[j, :] = cur

        s3, adj = self.gc(pre_feature, cur_feature, adj1)

        return s0, s1, s2, s3, adj1, adj
