# -*- coding: utf-8 -*-
# @File    : FuckUpNet.py
# @Author  : Peizhao Li
# @Contact : lipeizhao1997@gmail.com 
# @Date    : 2018/10/13

import torch
from torch import nn
import torch.nn.functional as F
from EmbeddingNet import EmbeddingNet
from EmbeddingNet import EmbeddingNet_train
from GCN import GCN
from GCN import fuckupGCN


class FuckUpNet(nn.Module):

    def __init__(self):
        super(FuckUpNet, self).__init__()
        self.embeddingnet = EmbeddingNet()
        self.gc = GCN(planes=144)

    def forward(self, pre_crop, cur_crop, pre_motion, cur_motion):
        matrix = torch.zeros(len(pre_crop), len(cur_crop)).cuda()
        pre_feature = torch.zeros(len(pre_crop), 144).cuda()
        cur_feature = torch.zeros(len(cur_crop), 144).cuda()

        cur_num = len(cur_crop)
        pre_num = len(pre_crop)

        for i in range(pre_num):
            pre_crop_ = pre_crop[i].cuda().unsqueeze_(dim=0)
            pre_motion_ = pre_motion[i].cuda().unsqueeze_(dim=0)
            for j in range(cur_num):
                cur_crop_ = cur_crop[j].cuda().unsqueeze_(dim=0)
                cur_motion_ = cur_motion[j].cuda().unsqueeze_(dim=0)
                score, pre, cur = self.embeddingnet(pre_crop_, cur_crop_, pre_motion_, cur_motion_)
                matrix[i, j] = score[0, 1]
                pre_feature[i, :] = pre
                cur_feature[j, :] = cur

        # matrix = self.gc(pre_feature, cur_feature, matrix)

        return matrix


class fuckupnet(nn.Module):

    def __init__(self):
        super(fuckupnet, self).__init__()
        self.embeddingnet = EmbeddingNet_train()
        self.gc = fuckupGCN(planes=144)

    def forward(self, pre_crop, cur_crop, pre_motion, cur_motion):
        matrix = torch.zeros(len(pre_crop), len(cur_crop)).cuda()
        score0 = torch.zeros(len(pre_crop) * len(cur_crop), 2).cuda()
        pre_feature = torch.zeros(len(pre_crop), 144).cuda()
        cur_feature = torch.zeros(len(cur_crop), 144).cuda()

        cur_num = len(cur_crop)
        pre_num = len(pre_crop)

        for i in range(pre_num):
            pre_crop_ = pre_crop[i].cuda().unsqueeze_(dim=0)
            pre_motion_ = pre_motion[i].cuda().unsqueeze_(dim=0)
            for j in range(cur_num):
                cur_crop_ = cur_crop[j].cuda().unsqueeze_(dim=0)
                cur_motion_ = cur_motion[j].cuda().unsqueeze_(dim=0)
                score, pre, cur = self.embeddingnet(pre_crop_, cur_crop_, pre_motion_, cur_motion_)
                score = score.squeeze_()
                score0[i * cur_num + j] = score
                matrix[i, j] = score[1]
                pre_feature[i, :] = pre
                cur_feature[j, :] = cur

        # score1, score2, matrix = self.gc(pre_feature, cur_feature, matrix)

        return score0, matrix


class embnet(nn.Module):

    def __init__(self):
        super(embnet, self).__init__()
        self.embeddingnet = EmbeddingNet_train()

    def forward(self, pre_crop, cur_crop, pre_motion, cur_motion):
        matrix = torch.zeros(len(pre_crop), len(cur_crop)).cuda()
        pre_feature = torch.zeros(len(pre_crop), 144).cuda()
        cur_feature = torch.zeros(len(cur_crop), 144).cuda()

        cur_num = len(cur_crop)
        pre_num = len(pre_crop)

        for i in range(pre_num):
            pre_crop_ = pre_crop[i].cuda().unsqueeze_(dim=0)
            pre_motion_ = pre_motion[i].cuda().unsqueeze_(dim=0)
            for j in range(cur_num):
                cur_crop_ = cur_crop[j].cuda().unsqueeze_(dim=0)
                cur_motion_ = cur_motion[j].cuda().unsqueeze_(dim=0)
                score, pre, cur = self.embeddingnet(pre_crop_, cur_crop_, pre_motion_, cur_motion_)
                score = score.squeeze_()
                matrix[i, j] = score[1]
                pre_feature[i, :] = pre
                cur_feature[j, :] = cur

        return matrix, pre_feature, cur_feature


class uninet(nn.Module):

    def __init__(self):
        super(uninet, self).__init__()
        self.embnet = embnet()
        self.gc = fuckupGCN(planes=144)

    def forward(self, pre_crop, cur_crop, pre_motion, cur_motion):
        matrix, pre, cur = self.embnet(pre_crop, cur_crop, pre_motion, cur_motion)
        score1, score2, matrix = self.gc(pre, cur, matrix)

        return matrix, score1, score2
