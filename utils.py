# -*- coding: utf-8 -*-
# @File    : utils.py
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# @Date    : 2018/9/27

import yaml, torch, time, os
from easydict import EasyDict as edict
import numpy as np


def Config(filename):
    listfile1 = open(filename, 'r')
    listfile2 = open(filename, 'r')
    parser = edict(yaml.load(listfile1))
    settings_show = listfile2.read().splitlines()
    return parser, settings_show


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    multiple = 1
    for (gamma, step) in zip(gammas, schedule):
        if (epoch == step):
            multiple = gamma
            break
    all_lrs = []
    for param_group in optimizer.param_groups:
        param_group['lr'] = multiple * param_group['lr']
        all_lrs.append(param_group['lr'])
    return set(all_lrs)


def print_log(print_string, log, true_string=None):
    print("{}".format(print_string))
    if true_string is not None:
        print_string = true_string
    if log is not None:
        log.write('{}\n'.format(print_string))
        log.flush()


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


def time_for_file():
    ISOTIMEFORMAT = '%h-%d-at-%H-%M'
    return '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def extract_label(matrix):
    index = np.argwhere(matrix == 1)
    target = index[:, 1]
    target = torch.from_numpy(target).cuda()

    return target


def matrix_loss(matrix, gt_matrix, criterion_CE, criterion_MSE):
    index_row_match = np.where([np.sum(gt_matrix, axis=1) == 1])[1]
    index_col_match = np.where([np.sum(gt_matrix, axis=0) == 1])[1]
    index_row_miss = np.where([np.sum(gt_matrix, axis=1) == 0])[1]
    index_col_miss = np.where([np.sum(gt_matrix, axis=0) == 0])[1]

    gt_matrix_row_match = np.take(gt_matrix, index_row_match, axis=0)
    gt_matrix_col_match = np.take(gt_matrix.transpose(), index_col_match, axis=0)

    index_row_match = torch.from_numpy(index_row_match).cuda()
    index_col_match = torch.from_numpy(index_col_match).cuda()

    matrix_row_match = torch.index_select(matrix, dim=0, index=index_row_match)
    matrix_col_match = torch.index_select(matrix.t(), dim=0, index=index_col_match)

    label_row_CE = extract_label(gt_matrix_row_match)
    label_col_CE = extract_label(gt_matrix_col_match)

    loss = criterion_CE(matrix_row_match, label_row_CE)
    loss += criterion_CE(matrix_col_match, label_col_CE)

    if index_row_miss.size != 0:
        index_row_miss = torch.from_numpy(index_row_miss).cuda()
        matrix_row_miss = torch.index_select(matrix, dim=0, index=index_row_miss)
        loss += criterion_MSE(torch.sigmoid(matrix_row_miss), torch.zeros_like(matrix_row_miss))

    if index_col_miss.size != 0:
        index_col_miss = torch.from_numpy(index_col_miss).cuda()
        matrix_col_miss = torch.index_select(matrix.t(), dim=0, index=index_col_miss)
        loss += criterion_MSE(torch.sigmoid(matrix_col_miss), torch.zeros_like(matrix_col_miss))

    return loss


def accuracy(input, target):
    assert input.size() == target.size()

    input[input < 0] = 0
    input[input > 0] = 1
    batch_size = input.size(0)
    pos_size = torch.sum(target)

    dis = input.sub(target)
    wrong = torch.sum(torch.abs(dis))
    acc = (batch_size - wrong.item()) / batch_size

    index = torch.nonzero(target)
    input_pos = torch.sum(input[index])
    acc_pos = input_pos.item() / pos_size

    return acc, acc_pos


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
