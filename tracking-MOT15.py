# -*- coding: utf-8 -*-
# @File    : tracking-MOT15.py
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com 
# @Date    : 2018/11/14

from Test import TestGenerator
from utils import *
from tracking_utils import *
from mkvideo_MOT15 import MakeVideo

np.set_printoptions(precision=2, suppress=True)


def main(info, timer):
    "-------------------------------- initialize --------------------------------"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    parser, settings_show = Config("setting/{}_config.yml".format(info))
    for idx, data in enumerate(settings_show):
        print(data)
    detection = MakeCell(np.loadtxt("test-MOT15/{}/det.txt".format(info)))
    manual_init = MakeCell(np.loadtxt("test-MOT15/{}/res.txt".format(info)))
    StartFrame = len(manual_init)
    TotalFrame = len(detection)
    tracker_ = tracker(ID_assign_init=parser.ID_assign_init, ID_birth_init=parser.ID_birth_init,
                       DeathBufferLength=parser.DeathBufferLength, BirthBufferLength=parser.BirthBufferLength,
                       DeathCount=parser.DeathCount, BirthCount=parser.BirthCount, Threshold=parser.Threshold,
                       Distance=parser.Distance, BoxRation=parser.BoxRation, FrameWidth=parser.FrameWidth,
                       FrameHeight=parser.FrameHeight, PredictThreshold=parser.PredictThreshold)
    PrevData = manual_init[-1]
    PPrevData = manual_init[-2]
    PPPrevData = manual_init[-3]
    PPPPrevData = manual_init[-4]
    PPPPPrevData = manual_init[-5]

    res_path = "test-MOT15/{}/{}.txt".format(info, time_for_file())
    res_init = np.loadtxt("test-MOT15/{}/res.txt".format(info))
    with open(res_path, "w") as res:
        np.savetxt(res, res_init, fmt='%12.3f')
        res.close()

    generator = TestGenerator(res_path, info)
    BirthLog = [[], [], []]  # prev frame, negative ID, assign ID
    DeathLog = [[], []]  # death frame, death ID
    "----------------------------------------------------------------------------"

    print("############### Start Tracking ###############")
    for frame in range(StartFrame, TotalFrame):
        print ("-----------------------> Start Tracking Frame %d" % (frame + 1))
        CurData = detection[frame]

        PrevIDs = PrevData[:, 1].copy()
        assert PrevIDs[PrevIDs == 0].size == 0
        CurIDs = CurData[:, 1].copy()
        assert CurIDs.max() == -1 and CurIDs.min() == -1

        tik = time.time()
        Amatrix = generator(SeqID=0, frame=frame)
        Amatrix = Amatrix.cpu().numpy()
        assert Amatrix.shape[0] == PrevIDs.shape[0] and Amatrix.shape[1] == CurIDs.shape[0]

        PrevData, PPrevData, PPPrevData, PPPPrevData, PPPPPrevData, BirthLog, DeathLog = tracker_(Amatrix=Amatrix,
                                                                                                  PrevIDs=PrevIDs,
                                                                                                  CurData=CurData,
                                                                                                  PrevData=PrevData,
                                                                                                  PPrevData=PPrevData,
                                                                                                  PPPrevData=PPPrevData,
                                                                                                  PPPPrevData=PPPPrevData,
                                                                                                  PPPPPrevData=PPPPPrevData,
                                                                                                  BirthLog=BirthLog,
                                                                                                  DeathLog=DeathLog)
        tok = time.time()
        timer.sum(tok - tik)

        with open(res_path, "a") as res:
            np.savetxt(res, PrevData, fmt="%12.3f")
            res.close()

        print ("-----------------------> Finish Tracking Frame %d\n" % (frame + 1))
    print("############### Finish Tracking ###############\n")

    assert len(BirthLog[0]) == len(BirthLog[1]) == len(BirthLog[2])
    assert len(DeathLog[0]) == len(DeathLog[1])

    res_data = np.loadtxt(res_path)

    print("cleaning birth...")
    for birth in range(len(BirthLog[0])):
        frame = BirthLog[0][birth]
        ID_index = np.where(res_data[:, 1] == BirthLog[1][birth])
        assign_ID = BirthLog[2][birth]
        for i in range(parser.BirthCount):
            frame_ = frame - i
            frame_index = np.where(res_data[:, 0] == frame_)
            index = np.intersect1d(frame_index, ID_index)
            res_data[index, 1] = assign_ID

    print("cleaning death...")
    for death in range(len(DeathLog[0])):
        frame = DeathLog[0][death]
        ID_index = np.where(res_data[:, 1] == DeathLog[1][death])
        for i in range(parser.DeathCount - 3):
            frame_ = frame - i
            frame_index = np.where(res_data[:, 0] == frame_)
            index = np.intersect1d(frame_index, ID_index)
            res_data[index, 1] = -1

    print("cleaning death sp...")
    DeathBuffer = tracker_.DeathBuffer
    death_sp_log = np.intersect1d(np.where(DeathBuffer > 3)[0], np.where(DeathBuffer < parser.DeathCount)[0])
    for death_sp in range(death_sp_log.shape[0]):
        frame = TotalFrame
        ID_index = np.where(res_data[:, 1] == death_sp_log[death_sp])
        for i in range(int(DeathBuffer[death_sp_log[death_sp]])):
            frame_ = frame - i
            frame_index = np.where(res_data[:, 0] == frame_)
            index = np.intersect1d(frame_index, ID_index)
            res_data[index, 1] = -1

    np.savetxt(res_path, res_data, fmt="%12.3f")

    MakeVideo(res_path, info, parser.fps, parser.FrameWidth, parser.FrameHeight)


if __name__ == "__main__":
    # seq = ["ADL-Rundle-1", "ADL-Rundle-3", "AVG-TownCentre", "ETH-Crossing", "ETH-Jelmoli", "ETH-Linthescher", "KITTI-16",
    #        "KITTI-19", "PETS09-S2L2", "TUD-Crossing", "Venice-1"]
    seq = ["PETS09-S2L2"]

    timer = timer()
    for s in range(len(seq)):
        main(seq[s], timer)
    print("total time: {} second".format(timer()))
