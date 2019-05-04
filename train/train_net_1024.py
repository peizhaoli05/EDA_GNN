# -*- coding: utf-8 -*-
# @File    : train_net_1024.py
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# @Date    : 2018/10/24

import os.path as osp

from model import net_1024
from utils import *


def train(parser, generator, log, log_path):
    # print("training net_1024\n")
    # model = net_1024.net_1024()

    print("training final\n")
    model = net_1024.net_1024()

    "----------------- pretrained model loading -----------------"
    # print("loading pretrained model")
    # checkpoint = torch.load("/home/lallazhao/MOT/result/Oct-25-at-02-17-net_1024/net_1024_88.4.pth")
    # model.load_state_dict(checkpoint["state_dict"])
    "------------------------------------------------------------"

    model = model.cuda()
    net_param_dict = model.parameters()

    weight = torch.Tensor([10])
    criterion_BCE = torch.nn.BCEWithLogitsLoss(pos_weight=weight).cuda()
    criterion_CE = torch.nn.CrossEntropyLoss().cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()

    if parser.optimizer == "SGD":
        optimizer = torch.optim.SGD(net_param_dict, lr=parser.learning_rate,
                                    momentum=parser.momentum, weight_decay=parser.decay, nesterov=True)
    elif parser.optimizer == "Adam":
        optimizer = torch.optim.Adam(net_param_dict, lr=parser.learning_rate, weight_decay=parser.decay)
    elif parser.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(net_param_dict, lr=parser.learning_rate, weight_decay=parser.decay,
                                        momentum=parser.momentum)
    else:
        raise NotImplementedError

    # Main Training and Evaluation Loop
    start_time, epoch_time = time.time(), AverageMeter()

    Batch_time = AverageMeter()
    Loss = AverageMeter()
    Acc = AverageMeter()
    Acc_pos = AverageMeter()

    for epoch in range(parser.start_epoch, parser.epochs):
        all_lrs = adjust_learning_rate(optimizer, epoch, parser.gammas, parser.schedule)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (parser.epochs - epoch))

        # ----------------------------------- train for one epoch -----------------------------------
        batch_time, loss, acc, acc_pos = train_net_1024(model, generator, optimizer, criterion_BCE, criterion_CE,
                                                        criterion_MSE)

        Batch_time.update(batch_time)
        Loss.update(loss.item())
        Acc.update(acc)
        Acc_pos.update(acc_pos)

        if epoch % parser.print_freq == 0 or epoch == parser.epochs - 1:
            print_log('Epoch: [{:03d}/{:03d}]\t'
                      'Time {batch_time.val:5.2f} ({batch_time.avg:5.2f})\t'
                      'Loss {loss.val:6.3f} ({loss.avg:6.3f})\t'
                      "Acc {acc.val:6.3f} ({acc.avg:6.3f})\t"
                      "Acc_pos {acc_pos.val:6.3f} ({acc_pos.avg:6.3f})\t".format(
                epoch, parser.epochs, batch_time=Batch_time, loss=Loss, acc=Acc, acc_pos=Acc_pos), log)

            Batch_time = AverageMeter()
            Loss = AverageMeter()

        if (epoch in parser.schedule):
            print_log("------------------- adjust learning rate -------------------", log)
        # -------------------------------------------------------------------------------------------

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    if parser.save_model:
        save_file_path = osp.join(log_path, "net_1024.pth")
        states = {
            "state_dict": model.state_dict(),
        }
        torch.save(states, save_file_path)


def train_net_1024(model, generator, optimizer, criterion_BCE, criterion_CE, criterion_MSE):
    # switch to train mode
    model.train()

    cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix = generator()
    assert len(cur_crop) == len(cur_motion)
    assert len(pre_crop) == len(pre_motion)

    target = torch.from_numpy(gt_matrix).cuda().float().view(-1)

    end = time.time()

    s0, s1, s2, s3, adj1, adj = model(pre_crop, cur_crop, pre_motion, cur_motion)
    loss = criterion_BCE(s0, target)
    loss += criterion_BCE(s1, target)
    loss += criterion_BCE(s2, target)
    loss += criterion_BCE(s3, target)
    loss += matrix_loss(adj1, gt_matrix, criterion_CE, criterion_MSE)
    loss += matrix_loss(adj, gt_matrix, criterion_CE, criterion_MSE)

    # s0, s3, adj = model(pre_crop, cur_crop)
    # loss = criterion_BCE(s0, target)
    # loss = criterion_BCE(s3, target)
    # loss += matrix_loss(adj1, gt_matrix, criterion_CE, criterion_MSE)
    # loss += matrix_loss(adj, gt_matrix, criterion_CE, criterion_MSE)

    acc, acc_pos = accuracy(s3.clone(), target.clone())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batch_time = time.time() - end

    return batch_time, loss, acc, acc_pos
