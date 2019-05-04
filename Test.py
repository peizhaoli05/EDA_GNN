# -*- coding: utf-8 -*-
# @File    : Test.py
# @Author  : Peizhao Li
# @Contact : lipeizhao1997@gmail.com 
# @Date    : 2018/10/6

import os
import os.path as osp
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import net_1024


def LoadImg(img_path):
    path = os.listdir(img_path)
    path.sort()
    imglist = []

    for i in range(len(path)):
        img = Image.open(osp.join(img_path, path[i]))
        imglist.append(img.copy())
        img.close()

    return imglist


def LoadModel(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval()

    return model


class VideoData(object):

    def __init__(self, info, res_path):
        # MOT17
        # self.img = LoadImg("MOT17/MOT17/test/MOT17-{}-{}/img1".format(info[0], info[1]))
        # self.det = np.loadtxt("test/MOT17-{}-{}/det.txt".format(info[0], info[1]))

        # MOT15
        self.img = LoadImg("MOT15/test/{}/img1".format(info))
        self.det = np.loadtxt("test-MOT15/{}/det.txt".format(info))

        self.res_path = res_path

        self.ImageWidth = self.img[0].size[0]
        self.ImageHeight = self.img[0].size[1]
        self.transforms = transforms.Compose([
            transforms.Resize((84, 32)),
            transforms.ToTensor()
        ])

    def DetData(self, frame):
        data = self.det[self.det[:, 0] == (frame + 1)]

        return data

    def PreData(self, frame):
        res = np.loadtxt(self.res_path)
        DataList = []
        for i in range(5):
            data = res[res[:, 0] == (frame + 1 - i)]
            DataList.append(data)

        return DataList

    def TotalFrame(self):
        return len(self.img)

    def CenterCoordinate(self, SingleLineData):
        x = (SingleLineData[2] + (SingleLineData[4] / 2)) / float(self.ImageWidth)
        y = (SingleLineData[3] + (SingleLineData[5] / 2)) / float(self.ImageHeight)

        return x, y

    def Appearance(self, data):
        appearance = []

        img = self.img[int(data[0, 0]) - 1]
        for i in range(data.shape[0]):
            crop = img.crop((int(data[i, 2]), int(data[i, 3]), int(data[i, 2]) + int(data[i, 4]),
                             int(data[i, 3]) + int(data[i, 5])))
            crop = self.transforms(crop)
            appearance.append(crop)

        return appearance

    def DetMotion(self, data):
        motion = []
        for i in range(data.shape[0]):
            coordinate = torch.zeros([2])
            coordinate[0], coordinate[1] = self.CenterCoordinate(data[i])
            motion.append(coordinate)

        return motion

    def PreMotion(self, DataTuple):
        motion = []
        nameless = DataTuple[0]
        for i in range(nameless.shape[0]):
            coordinate = torch.zeros([5, 2])
            identity = nameless[i, 1]
            coordinate[4, 0], coordinate[4, 1] = self.CenterCoordinate(nameless[i])
            # print(identity)

            for j in range(1, 5):
                unknown = DataTuple[j]
                if identity in unknown[:, 1]:
                    coordinate[4 - j, 0], coordinate[4 - j, 1] = self.CenterCoordinate(
                        unknown[unknown[:, 1] == identity].squeeze())
                else:
                    coordinate[4 - j, :] = coordinate[5 - j, :]

            motion.append(coordinate)

        return motion

    def GetID(self, data):
        id = []
        for i in range(data.shape[0]):
            id.append(data[i, 1].copy())

        return id

    def __call__(self, frame):
        assert frame >= 5 and frame < self.TotalFrame()
        det = self.DetData(frame)
        pre = self.PreData(frame - 1)
        det_crop = self.Appearance(det)
        pre_crop = self.Appearance(pre[0])
        det_motion = self.DetMotion(det)
        pre_motion = self.PreMotion(pre)
        pre_id = self.GetID(pre[0])

        return det_crop, det_motion, pre_crop, pre_motion, pre_id


class TestGenerator(object):

    def __init__(self, res_path, info):
        net = net_1024.net_1024()
        net_path = "SaveModel/net_1024_beta2.pth"
        print("------->  loading net_1024")
        self.net = LoadModel(net, net_path)

        self.sequence = []

        print("------->  initializing  MOT17-{}-{} ...".format(info[0], info[1]))
        self.sequence.append(VideoData(info, res_path))
        print("------->  initialize  MOT17-{}-{}  done".format(info[0], info[1]))

        self.vis_save_path = "test/visualize"

    def visualize(self, SeqID, frame, save_path=None):
        """

        :param seq_ID:
        :param frame:
        :param save_path:
        """
        if save_path is None:
            save_path = self.vis_save_path

        print("visualize sequence {}: frame {}".format(self.SequenceID[SeqID], frame + 1))
        print("video solution: {} {}".format(self.sequence[SeqID].ImageWidth, self.sequence[SeqID].ImageHeight))
        det_crop, det_motion, pre_crop, pre_motion, pre_id = self.sequence[SeqID](frame)

        for i in range(len(det_crop)):
            img = det_crop[i]
            img = transforms.functional.to_pil_image(img)
            img = transforms.functional.resize(img, (420, 160))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), "num: {}\ncoord: {:3.2f}, {:3.2f}".format(int(i), det_motion[i][0].item(),
                                                                        det_motion[i][1].item()), fill=(255, 0, 0))
            img.save(osp.join(save_path, "det_crop_{}.png".format(str(i).zfill(2))))

        for i in range(len(pre_crop)):
            img = pre_crop[i]
            img = transforms.functional.to_pil_image(img)
            img = transforms.functional.resize(img, (420, 160))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), "num: {}\nid: {}\ncoord: {:3.2f}, {:3.2f}".format(int(i), int(pre_id[i]),
                                                                                pre_motion[i][4, 0].item(),
                                                                                pre_motion[i][4, 1].item()),
                      fill=(255, 0, 0))
            img.save(osp.join(save_path, "pre_crop_{}.png".format(str(i).zfill(2))))

        np.savetxt(osp.join(save_path, "pre_id.txt"), np.array(pre_id).transpose(), fmt="%d")

    def __call__(self, SeqID, frame):
        # frame start with 5, exist frame start from 1
        sequence = self.sequence[SeqID]
        det_crop, det_motion, pre_crop, pre_motion, pre_id = sequence(frame)
        with torch.no_grad():
            s0, s1, s2, s3, adj1, adj = self.net(pre_crop, det_crop, pre_motion, det_motion)

        return adj
