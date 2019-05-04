# -*- coding: utf-8 -*-
# @File    : tracking_utils.py
# @Author  : Peizhao Li
# @Contact : peizhaoli05gmail.com 
# @Date    : 2018/11/2

import numpy as np


def MakeCell(data):
    cell = []
    frame_last = data[-1, 0]
    for i in range(1, int(frame_last) + 1):
        data_ = data[data[:, 0] == i]
        cell.append(data_.copy())

    return cell


class timer():

    def __init__(self):
        self.time = 0

    def sum(self, time):
        self.time += time

    def __call__(self):
        return int(self.time)


class ID_assign():

    def __init__(self, ID_init):
        self.ID = ID_init - 1

    def curID(self):
        return self.ID

    def __call__(self):
        self.ID += 1
        return self.ID


class ID_birth():

    def __init__(self, ID_init):
        self.ID = ID_init + 1

    def curID(self):
        return self.ID

    def __call__(self):
        self.ID -= 1
        return self.ID


class tracker():

    def __init__(self, ID_assign_init, ID_birth_init, DeathBufferLength, BirthBufferLength, DeathCount, BirthCount,
                 Threshold, Distance, BoxRation, FrameWidth, FrameHeight, PredictThreshold):
        self.ID_assign = ID_assign(ID_init=ID_assign_init)
        self.ID_birth = ID_birth(ID_init=ID_birth_init)
        self.DeathBuffer = np.zeros(DeathBufferLength)
        self.BirthBuffer = np.zeros(BirthBufferLength)
        self.DeathCount = DeathCount
        self.BirthCount = BirthCount
        self.Threshold = Threshold
        self.Distance = Distance
        self.BoxRation = BoxRation
        self.FrameWidth = float(FrameWidth)
        self.FrameHeight = float(FrameHeight)
        self.PredictThreshold = PredictThreshold

    def DistanceMeasure(self, PrevData_, CurData_):
        x_dis = (np.abs(PrevData_[2] - CurData_[2])) / self.FrameWidth
        y_dis = (np.abs(PrevData_[3] - CurData_[3])) / self.FrameHeight
        dis = x_dis + y_dis

        PrevBoxSize = PrevData_[4]
        CurBoxSize = CurData_[4]

        if dis <= self.Distance and CurBoxSize < PrevBoxSize * (1 + self.BoxRation) and CurBoxSize > PrevBoxSize * (
                1 - self.BoxRation):
            return True
        else:
            return False

    def CoordPrediction_v1(self, PrevData_, PPrevData_):
        if PPrevData_ is not None:
            x = 2 * PrevData_[2] - PPrevData_[2]
            y = 2 * PrevData_[3] - PPrevData_[3]
        else:
            x = PrevData_[2]
            y = PrevData_[3]

        return x, y

    def CoordPrediction_v2(self, PrevData_, PPrevData_):
        if PPrevData_ is not None:
            w = (PrevData_[4] + PPrevData_[4]) / 2.0
            h = (PrevData_[5] + PPrevData_[5]) / 2.0

            x0 = PPrevData_[2] + (PPrevData_[4] / 2.0)
            y0 = PPrevData_[3] + (PPrevData_[5] / 2.0)

            x1 = PrevData_[2] + (PrevData_[4] / 2.0)
            y1 = PrevData_[3] + (PrevData_[5] / 2.0)

            x_move = x1 - x0
            x_move = min(x_move, self.PredictThreshold * self.FrameWidth)
            x_move = max(x_move, -self.PredictThreshold * self.FrameWidth)

            y_move = y1 - y0
            y_move = min(y_move, self.PredictThreshold * self.FrameHeight)
            y_move = max(y_move, -self.PredictThreshold * self.FrameHeight)

            x2 = x1 + x_move
            y2 = y1 + y_move

            x = x2 - (w / 2.0)
            y = y2 - (h / 2.0)

        else:
            x = PrevData_[2]
            y = PrevData_[3]
            w = PrevData_[4]
            h = PrevData_[5]

        return x, y, w, h

    def CoordPrediction_v3(self, PrevData_, PPrevData_, PPPrevData_):
        if PPrevData_ is not None and PPPrevData_ is not None:
            w = (PPrevData_[4] + PPPrevData_[4]) / 2.0
            h = (PPrevData_[5] + PPPrevData_[5]) / 2.0

            x0 = PPPrevData_[2] + (PPPrevData_[4] / 2.0)
            y0 = PPPrevData_[3] + (PPPrevData_[5] / 2.0)

            x1 = PPrevData_[2] + (PPrevData_[4] / 2.0)
            y1 = PPrevData_[3] + (PPrevData_[5] / 2.0)

            x2 = 3 * x1 - 2 * x0
            y2 = 3 * y1 - 2 * y0

            x = x2 - (w / 2.0)
            y = y2 - (h / 2.0)

        else:
            x = PrevData_[2]
            y = PrevData_[3]
            w = PrevData_[4]
            h = PrevData_[5]

        return x, y, w, h

    def CoordPrediction_v4(self, PrevData_, PPrevData_, PPPrevData_, PPPPrevData_, PPPPPrevData_):
        w = PrevData_[4]
        h = PrevData_[5]

        x0 = PPPPPrevData_[2] + (PPPPPrevData_[4] / 2.0)
        y0 = PPPPPrevData_[3] + (PPPPPrevData_[5] / 2.0)

        x1 = PPPPrevData_[2] + (PPPPrevData_[4] / 2.0)
        y1 = PPPPrevData_[3] + (PPPPrevData_[5] / 2.0)

        x2 = PPPrevData_[2] + (PPPrevData_[4] / 2.0)
        y2 = PPPrevData_[3] + (PPPrevData_[5] / 2.0)

        x3 = PPrevData_[2] + (PPrevData_[4] / 2.0)
        y3 = PPrevData_[3] + (PPrevData_[5] / 2.0)

        x4 = PrevData_[2] + (PrevData_[4] / 2.0)
        y4 = PrevData_[3] + (PrevData_[5] / 2.0)

        x_move = ((x1 - x0) + (x2 - x1) + (x3 - x2) + (x4 - x3)) / 4.0
        y_move = ((y1 - y0) + (y2 - y1) + (y3 - y2) + (y4 - y3)) / 4.0

        x_move = min(x_move, self.PredictThreshold * self.FrameWidth)
        x_move = max(x_move, -self.PredictThreshold * self.FrameWidth)

        y_move = min(y_move, self.PredictThreshold * self.FrameHeight)
        y_move = max(y_move, -self.PredictThreshold * self.FrameHeight)

        x5 = x4 + x_move
        y5 = y4 + y_move

        x = x5 - (w / 2.0)
        y = y5 - (h / 2.0)

        return x, y, w, h

    def __call__(self, Amatrix, PrevIDs, CurData, PrevData, PPrevData, PPPrevData, PPPPrevData, PPPPPrevData, BirthLog,
                 DeathLog):
        PreRange = np.arange(Amatrix.shape[0])
        CurRange = np.arange(Amatrix.shape[1])
        PrevMatchIndex = []
        CurMatchIndex = []

        # step 1: match
        while Amatrix.max() > self.Threshold:
            PrevIndex, CurIndex = np.unravel_index(Amatrix.argmax(), Amatrix.shape)
            if self.DistanceMeasure(PrevData[PrevIndex], CurData[CurIndex]):
                PrevMatchIndex.append(PrevIndex.copy())
                CurMatchIndex.append(CurIndex.copy())
                prevID = int(PrevIDs[PrevIndex])

                # step 1.1: birth check
                if prevID < 0:
                    self.BirthBuffer[prevID] += 1
                    print("ID %d birth count %d" % (prevID, self.BirthBuffer[prevID]))

                    if self.BirthBuffer[prevID] == self.BirthCount:
                        CurData[CurIndex, 1] = self.ID_assign()
                        BirthLog[0].append(PrevData[PrevIndex, 0])
                        BirthLog[1].append(prevID)
                        BirthLog[2].append(CurData[CurIndex, 1])
                        print("---> New ID %d assigned to index %d" % (CurData[CurIndex, 1], CurIndex))
                    else:
                        CurData[CurIndex, 1] = prevID

                # step 1.2: match
                else:
                    # step 1.2.1: buffer clean
                    self.DeathBuffer[prevID] = 0

                    # step 1.2.2: copy ID
                    CurData[CurIndex, 1] = prevID
                    print("ID %d passed from index %d to index %d" % (prevID, PrevIndex, CurIndex))

                Amatrix[PrevIndex, :] = self.Threshold
                Amatrix[:, CurIndex] = self.Threshold
            else:
                Amatrix[PrevIndex, CurIndex] = self.Threshold

        # step 2: find mismatch
        DeathIndex = np.setxor1d(np.array(PrevMatchIndex), PreRange).astype(int)
        BirthIndex = np.setxor1d(np.array(CurMatchIndex), CurRange).astype(int)
        print ("-----------------------> Birth and Death")
        print("DeathIndex: {}".format(DeathIndex))
        print("BirthIndex: {}".format(BirthIndex))

        # step 3: death process
        for i in range(len(DeathIndex)):
            deathID = int(PrevIDs[DeathIndex[i]])
            if deathID < 0:
                pass
            else:
                self.DeathBuffer[deathID] += 1
                print("ID %d death count %d" % (deathID, self.DeathBuffer[deathID]))

                # step 3.1: terminate check
                if self.DeathBuffer[deathID] == self.DeathCount:
                    DeathLog[0].append(PrevData[DeathIndex[i], 0])
                    DeathLog[1].append(deathID)
                    print("terminate %d" % deathID)

                # step 3.2: death prediction
                else:
                    PrevData_ = PrevData[PrevData[:, 1] == deathID].squeeze()
                    if deathID in PPrevData[:, 1]:
                        PPrevData_ = PPrevData[PPrevData[:, 1] == deathID].squeeze()
                    else:
                        PPrevData_ = PrevData_
                    if deathID in PPPrevData[:, 1]:
                        PPPrevData_ = PPPrevData[PPPrevData[:, 1] == deathID].squeeze()
                    else:
                        PPPrevData_ = PPrevData_
                    if deathID in PPPPrevData[:, 1]:
                        PPPPrevData_ = PPPPrevData[PPPPrevData[:, 1] == deathID].squeeze()
                    else:
                        PPPPrevData_ = PPPrevData_
                    if deathID in PPPPPrevData[:, 1]:
                        PPPPPrevData_ = PPPPPrevData[PPPPPrevData[:, 1] == deathID].squeeze()
                    else:
                        PPPPPrevData_ = PPPPrevData_
                    DeathData = PrevData[DeathIndex[i]].copy()

                    CoordPrediction = self.CoordPrediction_v4(PrevData_, PPrevData_, PPPrevData_, PPPPrevData_,
                                                              PPPPPrevData_)
                    DeathData[2], DeathData[3], DeathData[4], DeathData[5] = CoordPrediction[0], CoordPrediction[1], \
                                                                             CoordPrediction[2], CoordPrediction[3]

                    DeathData[0] += 1  # frame update
                    CurData = np.concatenate((CurData, DeathData.reshape(1, -1)))
                    print("ID %d coordinates predicted" % deathID)

        # step 4: birth process:
        for j in range(len(BirthIndex)):
            CurData[BirthIndex[j], 1] = self.ID_birth()
            print("Pseudo ID %d assigned to index %d" % (CurData[BirthIndex[j], 1], BirthIndex[j]))

        assert self.DeathBuffer.max() <= self.DeathCount
        assert self.BirthBuffer.max() <= self.BirthCount

        print("-----------------------> ID info")
        print("ID up to %d" % self.ID_assign.curID())
        print("Pseudo ID up to %d" % self.ID_birth.curID())

        return CurData, PrevData, PPrevData, PPPrevData, PPPPrevData, BirthLog, DeathLog
