# -*- coding: utf-8 -*-
# @File    : TestGenerate.py
# @Author  : Peizhao Li
# @Contact : peizhaoli05gmail.com 
# @Date    : 2018/10/30

from Test import TestGenerator
import scipy.io as sio
from utils import *
import os.path as osp

np.set_printoptions(precision=2, suppress=True)


def MakeCell(data):
    cell = []
    frame_last = data[-1, 0]
    for i in range(1, int(frame_last) + 1):
        data_ = data[data[:, 0] == i]
        cell.append(data_.copy())

    return cell


def TrackletRotation(newFrame, oldTracklet):
    # This function do the tracklet rotation in the prev_frame, happen at each frame preparing the input to the model
    # ---> oldTracklet is a 17-dim tuple
    # ---> newFrame is a 7-dim tuple

    # First, rotate the oldTracklet, i.e. split the oldTracklet into 7-dim head and 10-dim tail
    # rotate the tail by poping the last two, and insert(0) in the X,Y just tracked, i.e. newFrame[2,3]
    # Second, put the newFrame X, Y into the odlTracklet 2 & 3dim
    head = list(oldTracklet[:7])
    tail = list(oldTracklet[7:])
    for i in range(2):
        tail.pop()
        tail.insert(0, newFrame[3 - i])
        head[5 - i] = newFrame[5 - i]
        head[3 - i] = newFrame[3 - i]

    output = head + tail
    return output


def DistanceMeasure(prev_coord, current_coord, FrameW, FrameH):
    x_dist = abs(prev_coord[0] - current_coord[0])
    y_dist = abs(prev_coord[1] - current_coord[1])

    x_dist = float(x_dist) / float(FrameW)
    y_dist = float(y_dist) / float(FrameH)

    rst = x_dist + y_dist

    return rst


def Tracker(Amatrix, Prev_frame, PrevIDs, CurrentDRs, BirthBuffer, DeathBuffer, IDnow, FrameW, FrameH, NIDnow):
    '''
    This Tracker function output the CurrentDRs tensor as the thing to be wrote into the txt;
    CurrentDRs input as all TrID = -1;
    Then:
    ---> 1-to-1, write the trajID into the CurrentDRs correspondingly;
    ---> Birth, intitialize a new temporal ID in the birth buffer for that DR;         ---> Need a Birth Buffer: buffer is a list of tuples
    ---> Death, put the dead trajIDs into the death list;                              ---> Nedd a Death Buffer: buffer is a list of tuples
    '''

    '''
    ---> Trajectories: Len = 5, a list of 7 + 2 * 5 = 17-dim tuple: [FrID, TrID, X, Y, W, H, X1, Y1, X2, Y2, ...., X5, Y5], W & H remains; just trajectories in the prev_frame
    ---> PrevIDs: the IDs corresponds to the Trajectories, who index the IDs in the AMatrix rows
    ---> CurrentDRs:            a list of 7-dim tuple: [FrID, -1, X, Y, W, H]
    ---> BirthBuffer:           a list of 7-dim tuple: [FrID, 0, X, Y, W, H]
    ---> DeathBuffer:           a list of 7-dim tuple: [FrID, 'D', X, Y, W, H] for flags, use 'DD', 'DDD, etc

    RETURN ---> output, which is the ID assigned CurrentDRs
    '''

    prev_num = Amatrix.shape[1]  # Amatrix.shape[0] is the batch size
    next_num = Amatrix.shape[2]

    DROccupiedIndexes = []
    ConfirmedBirthRowID = []
    ToDo_DoomedTrajID = []  # Confirmed Death Trajs to be taken out of prev_frame after each Tracker() iteration
    Fail2ConfirmedBirthKill = []
    '''
    The New Reading Matrix Logic
    '''
    if 'New Reading Logic for the AMatrix':
        '''
        set a configurable threshold params, reading by iteratively localizing the largest item in the matrix,
        Then finish an association, set the associated row and column to a very small negative value,
        Then repeat, until the max value in the matrix is below Threshold, then the left un-associated left &
        column to reason death & birth
        '''
        Th = 0
        dist_Th = 0.05
        DeathRows = [i for i in range(prev_num)]
        BirthCols = [j for j in range(next_num)]

        # Start the main loop here, may never reach the upper bound prev_num * next_num
        for i in range(prev_num * next_num):
            # print '-------------------------- i = %d'%i
            # compute the row and column index for the max value in a matrix
            # As AMatrix is a list of tuples, then has to find the max value in each list (row)
            # Then find the max column index in that row
            for k in range(prev_num):
                # print '--------------------------- prev_num %d'%prev_num
                # print '--------------------------- next_num %d'%next_num
                # print '------------------------------ k = %d'%k
                row_maxValue = []  # the index of row_maxValue corresponds to the row index of AMatrix
                [row_maxValue.append(max(Amatrix[0, j])) for j in range(prev_num)]
                max_rowValue = max(row_maxValue)

                # 1-to-1 Associations
                if max_rowValue > Th:
                    '''
                    Cases of 1-to-1 Associations
                    '''
                    max_rowIndex = row_maxValue.index(max_rowValue)
                    max_colIndex = list(Amatrix[0, max_rowIndex]).index(max_rowValue)

                    # Mark these associated row and col out of the DeathRows and BirthCols
                    DeathRows.remove(max_rowIndex)
                    BirthCols.remove(max_colIndex)

                    print("Cases of 1-to-1 Association, the selected max row and col index:")
                    print [max_rowIndex, max_colIndex]

                    print("DeathRows:")
                    print DeathRows

                    print("BirthCols:")
                    print BirthCols

                    '''
                    # ----------------------------- Case 1: Normal 1-to-1 association
                    '''
                    if PrevIDs[max_rowIndex] > -10 and PrevIDs[max_rowIndex] != -1:
                        associated_DrIndex = max_colIndex

                        '''
                        ------- ------- ------- Distance Threshold
                        '''
                        prev_wh = Prev_frame[max_rowIndex][2:4]
                        current_wh = CurrentDRs[associated_DrIndex][2:4]
                        dist1 = DistanceMeasure(prev_wh, current_wh, FrameW, FrameH)

                        if dist1 <= dist_Th:
                            CurrentDRs[associated_DrIndex][1] = Prev_frame[max_rowIndex][1]
                            print(
                                     "Normal 1-to-1 Association at %d row, traj ID paased onto next frame %d DR_Index, is %d") % (
                                     max_rowIndex, associated_DrIndex, CurrentDRs[associated_DrIndex][1])
                            # Check if this association revive anybody in the DeathBuffer. i.e. if the associated ID is someone in the DeathBuffer, then it is revived, removed from DeathBuffer.
                            # the just associated trajID here is Prev_frame[i][1]
                            allTrajIDInDeathBuffer = [DeathBuffer[i][0] for i in range(len(DeathBuffer))]
                            if Prev_frame[k][1] in allTrajIDInDeathBuffer:
                                reviveTrajIndex = allTrajIDInDeathBuffer.index(Prev_frame[k][1])
                                revivedID = DeathBuffer[reviveTrajIndex][0]
                                print 'Trajectory %d is about to be revived from the DeathBuffer' % revivedID
                                DeathBuffer.remove(DeathBuffer[reviveTrajIndex])

                    '''
                    # --------------------------- Case 2: Birth confirmation 1-to-1 association
                    '''
                    if PrevIDs[max_rowIndex] <= -10:
                        print 'Birth confirmation 1-to-1 Association at %d row' % max_rowIndex
                        associated_DrIndex = max_colIndex

                        '''
                            ------- ------- ------- Distance Threshold
                        '''
                        prev_wh = Prev_frame[max_rowIndex][2:4]
                        current_wh = CurrentDRs[associated_DrIndex][2:4]
                        dist2 = DistanceMeasure(prev_wh, current_wh, FrameW, FrameH)

                        if dist2 <= dist_Th:
                            CurrentDRs[associated_DrIndex][1] = IDnow
                            print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  A New Target id as %d is generated $$$$$$$$$$$$$$$$$$$$$' % IDnow
                            IDnow = IDnow + 1

                            ConfirmedBirthRowID.append(PrevIDs[max_rowIndex])

                        elif dist2 > dist_Th:
                            Fail2ConfirmedBirthKill.append(PrevIDs[max_rowIndex])
                        # Conservation Check here, once a column is taken, it cannot be taken again, so remove it from the AMatrix
                        if associated_DrIndex in DROccupiedIndexes:
                            print 'Conservation Constraint violated at row %d when doing Birth confirmation 1-to-1 Association' % i
                        else:
                            DROccupiedIndexes.append(associated_DrIndex)

                    # change the row & col in this instance of 1-to-1 association to be all < Th
                    Amatrix[0][max_rowIndex, :] = Th
                    Amatrix[0][:, max_colIndex] = Th
                    # print 'Hi'

                # B & D for cols and rows
                if max_rowValue <= Th:
                    '''
                    The Rest un-associated rows are death, columns are birth
                    '''
                    if not DeathRows:
                        # print '---> Tracker: No Death at this association'
                        pass
                    if not BirthCols:
                        # print '---> Tracker: No Birth at this association'
                        pass

                    '''
                    ***************************  Death handles
                    # for a just dead target, if it is within the death window, just prolong it, do not report it to be death
                    # within the death windows frames.
                    '''
                    if DeathRows:
                        for t in range(len(DeathRows)):
                            if DeathRows[t] != -10000:
                                DeadTrajID = prev_frame[DeathRows[t]][1]

                                '''
                                If a Death is ID <= -10, then terminate it right away
                                The Birth if not confirmed in the next frame, then it should be terminated
                                This termination is handled by the death handle altogether
                                '''
                                if DeadTrajID <= -10:
                                    #DeathBuffer.append([DeadTrajID, 5])
                                    Fail2ConfirmedBirthKill.append(DeadTrajID)
                                    print '_+_+_+_+_+_+_+_+_+_+_+ In death handles, a temporal birth %d fail to confirm, about to be killed right away in Fail2ConfirmedBirthKill'%DeadTrajID

                                '''
                                Normal death, update the death flag, then invoke motion prediciton
                                '''
                                if DeadTrajID > -10:
                                    '''
                                     # Step 1: Check and update the death counter of this dead trajectory in the DeathBuffer
                                    '''
                                    allTrajIDInDeathBuffer = [DeathBuffer[i][0] for i in range(len(DeathBuffer))]

                                    if DeadTrajID == 1:
                                        print 'Gotcha'

                                    if DeadTrajID not in allTrajIDInDeathBuffer:
                                        DeathBuffer.append([DeadTrajID, 0])
                                    else:
                                        # find the index of the DeadID, update its death counter by + 1
                                        AllDeadIDs = [DeathBuffer[i][0] for i in range(len(DeathBuffer))]
                                        temp_index = AllDeadIDs.index(DeadTrajID)
                                        DeathBuffer[temp_index][1] += 1

                                    # Step 2: As this trajectory is dead, i.e. fail to associate to a DR to prolong itself in this frame, then use a Motion Prediction
                                    # to propagate into a new dummy bbox to add into the result CurrentDRs
                                    DeadTraj = Prev_frame[DeathRows[t]][:]

                                    DeathRows[t] = -10000

                                    # Do the motion prediction using the tracklets in DeadTraj to predict a [X, Y], together with the original W, H, form a new DR wit ID to add into the CurrentDRs
                                    '''
                                    -------- MOTION PREDICTION HERE ----------
                                    '''
                                    if 'Motion Prediction for Dummies':
                                        print '----------------------------------------------- A target %d is not associated in this frame Motion Prediction at play now.' % DeadTrajID
                                        temp_tracklet = list(DeadTraj[2:4]) + list(DeadTraj[
                                                                                   7:15])  # temp_tracklet is to be feed into the LSTM, a list of 5 pairs of (x, y)

                                        dummyWH = list(DeadTraj[4:6])
                                        # Establish a simple Linear model here, which is about to be substitued by a LSTM
                                        temp_x_seq = [temp_tracklet[i] for i in range(0, len(temp_tracklet), 2)]
                                        temp_y_seq = [temp_tracklet[j] for j in range(1, len(temp_tracklet), 2)]

                                        predicted_x = temp_x_seq[1] + temp_x_seq[1] - temp_x_seq[2]
                                        predicted_y = temp_y_seq[1] + temp_y_seq[1] - temp_y_seq[2]

                                        dummy = list(DeadTraj[:2]) + list([predicted_x, predicted_y]) + list(dummyWH) + [0]

                                        if predicted_x < 0 or predicted_y < 0:
                                            print 'Im the Storm'



                                        dummy[0] += 1
                                        dummy = np.array(dummy)

                                        CurrentDRs.append(dummy)

                            else:
                                pass

                    '''
                    ***************************  Birth handles
                    '''
                    if BirthCols:
                        for p in range(len(BirthCols)):
                            if BirthCols[p] != -10000:
                                Birth_DrIndex = BirthCols[p]
                                BirthCols[p] = -10000
                                CurrentDRs[Birth_DrIndex][1] = NIDnow  # The new birth is temporally IDed as 0, then moved into the BirthBuffer
                                print '_+_+_+_+_+_+_+_+_+_+_+_+_+_+ One temporal birth found, assigned NID is %d' %NIDnow
                                NIDnow = NIDnow  - 1
                                # BirthBuffer deprecated
                                # BirthBuffer.append(CurrentDRs[Birth_DrIndex][:])

                                print 'Birth at %d column' % Birth_DrIndex

                            else:
                                pass

                else:
                    break

    # -------------------------------------------------------- Auxiliary Per-Frame Operations ---------------------------------------------------- #
    '''
    -------------------------- Death Counter Check and dead Bbox termination ---------------------
    '''
    # Check Real Death (them IDs in the DeathBuffer that with death counter up to DeathWindow) for termination
    AllDeathCounter = [DeathBuffer[i][1] for i in range(len(DeathBuffer))]
    DoomedIDs = []
    DeathBufferRemoveOnSpot = []
    if DeathWindow in AllDeathCounter:
        for i in range(len(DeathBuffer)):
            if DeathBuffer[i][1] >= DeathWindow:
                print 'One Trajectory %d meets the DeathWindow = %d, and is about to be terminate' % (DeathBuffer[i][0], DeathWindow)
                DoomedIDs.append(DeathBuffer[i][0])
                #print ' ! ! ! ! ! ! ! ! ! ! ! ! ! Trajectory %d is being terminated:' % DeathBuffer[i][0]
                DeathBufferRemoveOnSpot.append([DeathBuffer[i][0], DeathWindow])
                #DeathBuffer.remove([DeathBuffer[i][0], DeathWindow])
                #print 'IM the Storm'

    # Remove those trajs that have met DeathWindow in this frame out of the DeathBuffer
    for i in range(len(DeathBufferRemoveOnSpot)):
        DeathBuffer.remove(DeathBufferRemoveOnSpot[i])

    #print 'line 312'

    # -------------------- How to terminate all the trajectories with the DoomedIDs: i.e. remove it from the prev_frame
    # DoomedTrajsIndex = []
    # for i in range(len(prev_frame)):
    #     if prev_frame[i][1] in DoomedIDs:
    #         DoomedTrajsIndex.append(i)

    # for i in range(len(DoomedTrajsIndex)):
    #     print ' ! ! ! ! ! ! ! ! ! ! ! ! ! Trajectory %d is being terminated:' % prev_frame[DoomedTrajsIndex[i]][1]
    #     ToDo_DoomedTrajID.append(DoomedTrajsIndex[i])
        # list(prev_frame).pop(DoomedTrajsIndex[i])

    # # -------------------- Conservation Check again
    # if DRavaliableIndexes != DRavaliableIndexes2:
    #     print 'Conservation check violated by checking the birth by scenario and birth by removing all occupied DRs.'

    '''
    Post-Tracking processing, do the rotation to get a 17_dim output
    Now CurrentDRs hold 7-dim where the TrID has been updated
    Prolong & rotate the prev_frame with CurrentDRs by associating with TrID
    '''
    # ------------------------------- Case 1: 1-to-1 Association
    # --------------- Rotate & update the prev_frame
    for i in range(len(prev_frame)):
        for j in range(len(CurrentDRs)):
            if prev_frame[i][1] == CurrentDRs[j][1]:
                prev_frame[i] = TrackletRotation(CurrentDRs[j], prev_frame[i])

    # ------------------------------- Case 2: Birth Confirmation
    # ------------ Birth Confirmation now is in the CurrentDRs with a newly assigned ID, need to find them, and pad 0
    ConfirmedBirthProlonged = []
    allPrevIDs = [prev_frame[i][1] for i in range(len(prev_frame))]

    for k in range(len(CurrentDRs)):
        '''
        ---- The padding for the confirmed birth -----
        '''
        if CurrentDRs[k][1] not in allPrevIDs and CurrentDRs[k][1] > -10 and CurrentDRs[k][1] != -1:
            padding = np.zeros(2 * TrackletLen)
            for i in range(len(padding)):
                if i % 2 == 0:
                    padding[i] = CurrentDRs[k][2]
                if i % 2 != 0:
                    padding[i] = CurrentDRs[k][3]

            CurrentDRs[k] = np.concatenate((CurrentDRs[k], padding))

            ConfirmedBirthProlonged.append(CurrentDRs[k])

        '''
        ---- The padding for the newly birth --------
        '''
        if CurrentDRs[k][1] not in allPrevIDs and CurrentDRs[k][1] <= -10 and CurrentDRs[k][1] != -1:
            padding = np.zeros(2 * TrackletLen)
            CurrentDRs[k] = np.concatenate((CurrentDRs[k], padding))

            ConfirmedBirthProlonged.append(CurrentDRs[k])

    if ConfirmedBirthProlonged:
        # print 'ConfirmedBirthProlonged size:'
        # print len(ConfirmedBirthProlonged[0])
        output = np.concatenate((Prev_frame, ConfirmedBirthProlonged))
    else:
        output = Prev_frame

    return output, BirthBuffer, DeathBuffer, IDnow, ConfirmedBirthRowID, DoomedIDs, NIDnow, Fail2ConfirmedBirthKill


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    data_framewise = np.loadtxt("test/MOT17-01-SDP/det.txt")
    data_framewise = MakeCell(data_framewise)
    manual_init = np.loadtxt("test/MOT17-01-SDP/res.txt")
    manual_init = MakeCell(manual_init)

    batchSize = 1
    TrackletLen = 5
    DeathWindow = 3
    V_len = len(data_framewise)
    Bcount = 0
    Dcount = 0
    FrameWidth = 1920
    FrameHeight = 1080
    BirthBuffer = []
    DeathBuffer = []
    prev_frame = manual_init[4]
    IDnow = 20
    NIDnow = -10

    padding = np.zeros((prev_frame.shape[0], 2 * TrackletLen))
    prev_frame = np.concatenate((prev_frame, padding), axis=1)

    for i in range(prev_frame.shape[0]):
        identity = prev_frame[i, 1]
        for frame in range(5):
            data = manual_init[4 - frame]
            if identity in data[:, 1]:
                data_ = data[data[:, 1] == identity].squeeze()
                prev_frame[i, 7 + 2 * frame], prev_frame[i, 8 + 2 * frame] = data_[2], data_[3]
            else:
                prev_frame[i, 7 + 2 * frame], prev_frame[i, 8 + 2 * frame] = prev_frame[i, 5 + 2 * frame], prev_frame[
                    i, 6 + 2 * frame]

    prev_frame = list(prev_frame)
    PrevIDs = [prev_frame[i][1] for i in range(len(prev_frame))]

    res_path = "test/MOT17-01-SDP/{}.txt".format(time_for_file())
    log_path = "test/MOT17-01-SDP/log.log"
    buffer_path = "test/MOT17-01-SDP/buffer.txt"
    if not osp.exists(res_path):
        os.mknod(res_path)
    res_init = np.loadtxt("test/MOT17-01-SDP/res.txt")

    generator = TestGenerator(res_path, entirety=False)

    with open(res_path, "a") as txt:
        for i in range(len(res_init)):
            temp = np.array(res_init[i]).reshape([1, -1])
            np.savetxt(txt, temp[:, 0:7], fmt='%12.3f')

    "---------------------- start tracking ----------------------"
    for v in range(4, V_len - 1):
        # v denotes the previous frame and start with 0
        prev = v  # prev: 1~TotalFrame-1
        next = prev + 1
        next_frame = list(data_framewise[next])

        FrID4now = next_frame[0][0]

        print ("------------------------------------------> Tracking Frame %d" % FrID4now)

        CurrentDRs = next_frame
        PrevIDs = [prev_frame[i][1] for i in range(len(prev_frame))]

        print("---> MAIN: Input to Model --> CurrentDRs")
        temp2 = [CurrentDRs[i][1] for i in range(len(CurrentDRs))]
        print(temp2)

        print("---> MAIN: Input to Model --> prev_frame")
        temp1 = [prev_frame[i][1] for i in range(len(prev_frame))]
        print(temp1)

        Amatrix = generator(SeqID=0, frame=next)
        Amatrix = Amatrix.unsqueeze(dim=0).cpu().numpy()


        for i in range(len(CurrentDRs)):
            CurrentDRs[i][1] = -1

        '''
        Initialize the trajectory sequences by appending the trajectories with 0s in the back
        '''
        prev_FrID = prev  # This FrID is only used to load in the frames from file
        next_FrID = prev_FrID + 1

        # Update the FrID for prev_frame
        for i in range(len(prev_frame)):
            prev_frame[i][0] = FrID4now

        ToPlot, Bbuffer, Dbuffer, IDnow_out, ConfirmedBirthRowID_out, DoomedIDs_out, NIDnow_out, Fail2ConfirmedBirthKill_out = Tracker(
            Amatrix=Amatrix, Prev_frame=prev_frame, PrevIDs=PrevIDs, CurrentDRs=CurrentDRs, BirthBuffer=BirthBuffer,
            DeathBuffer=DeathBuffer, IDnow=IDnow, FrameW=FrameWidth, FrameH=FrameHeight, NIDnow= NIDnow)

        '''
        Also terminate the Doomed trajs out of the DeathBuffer
        '''
        # for i in range(len(Dbuffer)):
        #     if Dbuffer[i][0] in ToDo_DoomedTrajID_out:
        #         del Dbuffer[i]

        '''
        When it is negative
        '''
        for i in range(len(ToPlot)):
            if ToPlot[i][2] < 0:
                print 'Now negative X, Y'

        '''
        ------------------------------------------------------------- Clean out the doomed ID that is killed
        '''
        DoomedTraj2CleanIndex = []
        for i in range(len(ToPlot)):
            if ToPlot[i][1] in DoomedIDs_out:
                DoomedTraj2CleanIndex.append(i)

        DoomedTraj2CleanIndex.sort(reverse=True)
        for i in range(len(DoomedTraj2CleanIndex)):
            print '---> MAIN: _+_+_+_+_+_+_+_  Popping DoomedTraj ID is %d'%ToPlot[DoomedTraj2CleanIndex[i]][1]
            ToPlot = np.delete(ToPlot, DoomedTraj2CleanIndex[i], 0)

        '''
       ----------------------------------------------------- Clean out the temporal birth that have been confirmed in this frame
        '''
        ConfirmedBirthRowIndex = []
        for i in range(len(ToPlot)):
            if ToPlot[i][1] in ConfirmedBirthRowID_out:
                ConfirmedBirthRowIndex.append(i)

        ConfirmedBirthRowIndex.sort(reverse= True)
        for i in range(len(ConfirmedBirthRowIndex)):
            print("---> MAIN: _+_+_+_+_+_+_+_  Popping the temporal birth out when a birth is confirmed, ID is %d")%ToPlot[ConfirmedBirthRowIndex[i],1]
            ToPlot = np.delete(ToPlot, ConfirmedBirthRowIndex[i], axis=0)

        '''
        ---------------------------------------------------- Fail2ConfirmedBirthKill_out kill right away
        '''
        Fail2ConfirmedBirthKill_out_index = []
        for i in range(len(ToPlot)):
            if ToPlot[i][1] in Fail2ConfirmedBirthKill_out:
                Fail2ConfirmedBirthKill_out_index.append(i)

        Fail2ConfirmedBirthKill_out_index.sort(reverse=True)
        for i in range(len(Fail2ConfirmedBirthKill_out_index)):
            print '---> MAIN: _+_+_+_+_+_+_+_  ToPlot delete Fail2ConfirmedBirthKill, ID is %d'%ToPlot[Fail2ConfirmedBirthKill_out_index[i],1]
            ToPlot = np.delete(ToPlot, Fail2ConfirmedBirthKill_out_index[i], axis = 0)


        print("---> MAIN: Tracking output--> ToPlot")
        temp3 = [ToPlot[i][1] for i in range(len(ToPlot))]
        print temp3

        temp6 = [ToPlot[i][2] for i in range(len(ToPlot))]
        for i in range(len(temp6)):
            if temp6[i] <= 0:
                print 'Im the storm'

        print("---> MAIN: Tracking output--> DeathBuffer")
        temp1 = [Dbuffer[i] for i in range(len(Dbuffer))]
        print temp1

        # print("---> MAIN: Tracking output--> BirthBuffer")
        # temp4 = [BirthBuffer[i][1] for i in range(len(BirthBuffer))]
        # print temp4
        #
        print("---> MAIN: Tracking output--> DeathBuffer")
        temp4 = [DeathBuffer[i][0] for i in range(len(DeathBuffer))]
        temp5 = [DeathBuffer[i][1] for i in range(len(DeathBuffer))]
        print temp4,temp5


        BirthBuffer = Bbuffer
        DeathBuffer = Dbuffer
        IDnow = IDnow_out
        NIDnow = NIDnow_out
        prev_frame = ToPlot

        print("Writing")
        with open(res_path, "a") as txt:
            for i in range(len(ToPlot)):
                temp = np.array(ToPlot[i]).reshape([1, -1])
                np.savetxt(txt, temp[:, 0:7], fmt='%12.3f')

        with open(log_path, "a") as log:
            for i in range(len(ToPlot)):
                temp = np.array(ToPlot[i]).reshape([1, -1])
                np.savetxt(log, temp, fmt='%12.3f')

        with open(buffer_path, "a") as DeathBufferTXT:
            for i in range(len(DeathBuffer)):
                temp = np.array(DeathBuffer[i]).reshape([1, -1])
                np.savetxt(DeathBufferTXT, temp, fmt="%12.3f", delimiter=",")

    print("Finish Tracking Frame %d" % v)
    print("Current Initialized ID is up to: %d" % IDnow)
