# -*- coding: utf-8 -*-
# @File    : EmbeddingNet.py
# @Author  : Peizhao Li
# @Contact : lipeizhao1997@gmail.com 
# @Date    : 2018/10/21

import torch
from torch import nn
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(self, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :].clone()

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
        x = F.leaky_relu(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.leaky_relu(x)
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.leaky_relu(x)
        x = F.max_pool2d(self.conv4(x), 2)
        x = F.leaky_relu(x)

        x = F.avg_pool2d(x, kernel_size=(5, 2))
        x = x.view(x.size(0), -1)

        return x


class EmbeddingNet_legacy(nn.Module):

    def __init__(self):
        super(EmbeddingNet_legacy, self).__init__()
        self.outsize = 128 + 2

        self.ANet = ANet()
        self.lstm = LSTM(hidden_size=2)

        self.fc1 = nn.Linear(self.outsize, int(self.outsize / 2))
        self.fc2 = nn.Linear(int(self.outsize / 2), int(self.outsize / 4))
        self.fc3 = nn.Linear(int(self.outsize / 4), 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0.1)

        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.2)

    def forward(self, pre_crop, cur_crop, pre_coord, cur_coord):
        pre_crop = self.ANet(pre_crop)
        cur_crop = self.ANet(cur_crop)

        pre_coord = self.lstm(pre_coord)

        pre = torch.cat((pre_crop, pre_coord), dim=1)
        cur = torch.cat((cur_crop, cur_coord), dim=1)

        x = F.leaky_relu_(self.fc1(pre.add_(-cur)))
        x = self.drop1(x)
        x = F.leaky_relu_(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)

        return x, pre, cur


class EmbeddingNet_train(nn.Module):

    def __init__(self):
        super(EmbeddingNet_train, self).__init__()
        self.ANet = ANet()
        self.lstm = LSTM(hidden_size=16)
        self.fc = nn.Linear(2, 16)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0.1)

        self.crop_fc1 = nn.Linear(128, 64)
        self.crop_fc2 = nn.Linear(64, 32)
        self.crop_fc3 = nn.Linear(32, 2)

        self.coord_fc1 = nn.Linear(16, 8)
        self.coord_fc2 = nn.Linear(8, 2)

        self.com = nn.Linear(4, 2)

    def forward(self, pre_crop, cur_crop, pre_coord, cur_coord):
        pre_crop = self.ANet(pre_crop)
        cur_crop = self.ANet(cur_crop)
        pre_coord = self.lstm(pre_coord)
        cur_coord = self.fc(cur_coord)

        crop = F.leaky_relu(self.crop_fc1(pre_crop.add_(-cur_crop)))
        crop = F.leaky_relu(self.crop_fc2(crop))
        crop = self.crop_fc3(crop)

        coord = F.leaky_relu(self.coord_fc1(pre_coord.add_(-cur_coord)))
        coord = self.coord_fc2(coord)

        com = torch.cat((crop, coord), dim=1)
        com = self.com(com)

        pre_feature = torch.cat((pre_crop, pre_coord), dim=1)
        cur_feature = torch.cat((cur_crop, cur_coord), dim=1)

        return com, pre_feature, cur_feature


class EmbeddingNet(nn.Module):

    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.ANet = ANet()
        self.lstm = LSTM(hidden_size=16)
        self.fc = nn.Linear(2, 16)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0.1)

        self.crop_fc1 = nn.Linear(128, 64)
        self.crop_fc2 = nn.Linear(64, 32)
        self.crop_fc3 = nn.Linear(32, 2)

        self.coord_fc1 = nn.Linear(16, 8)
        self.coord_fc2 = nn.Linear(8, 2)

        self.com = nn.Linear(4, 2)

    def forward(self, pre_crop, cur_crop, pre_coord, cur_coord):
        pre_crop = self.ANet(pre_crop)
        cur_crop = self.ANet(cur_crop)
        pre_coord = self.lstm(pre_coord)
        cur_coord = self.fc(cur_coord)

        crop = F.leaky_relu(self.crop_fc1(pre_crop.add_(-cur_crop)))
        crop = F.leaky_relu(self.crop_fc2(crop))
        crop = self.crop_fc3(crop)

        coord = F.leaky_relu(self.coord_fc1(pre_coord.add_(-cur_coord)))
        coord = self.coord_fc2(coord)

        com = torch.cat((crop, coord), dim=1)
        com = self.com(com)

        pre_feature = torch.cat((pre_crop, pre_coord), dim=1)
        cur_feature = torch.cat((cur_crop, cur_coord), dim=1)

        return com, pre_feature, cur_feature
