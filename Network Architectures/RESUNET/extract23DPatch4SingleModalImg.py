'''
Target: Crop patches for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
for single input modality
Created in June, 2016
Author: Dong Nie
'''

import SimpleITK as sitk

from multiprocessing import Pool
import os, argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--how2normalize", type=int, default=6, help="how to normalize the data")
parser.add_argument("--noisy_file", type=str, default="", help="path to the noisy data in numpy format NxWxH")
parser.add_argument("--clear_file", type=str, default="", help="path to the clear data in numpy format NxWxH")
parser.add_argument("--save_folder", type=str, default="data/", help="path to save the data")
parser.add_argument("--split", type=str, default="train", help="train, dev or test")

global opt
opt = parser.parse_args()

# input patch size
d1 = 2
d2 = 64
d3 = 64
# output patch size
dFA = [d1, d2, d3]  # size of patches of input data
dSeg = [1, 64, 64]  # size of pathes of label data
# stride for extracting patches along the volume
step1 = 1
step2 = 8 
step3 = 8
step = [step1, step2, step3]

def extractPatch4OneSubject(matFA, matSeg, matMask, fileID, d, step, rate):
    eps = 5e-2
    [row, col, leng] = matFA.shape
    cubicCnt = 0
    estNum = 40000
    trainFA = np.zeros([estNum, 1, dFA[0], dFA[1], dFA[2]], dtype=np.float16)
    trainSeg = np.zeros([estNum, 1, dSeg[0], dSeg[1], dSeg[2]], dtype=np.float16)

    print('trainFA shape, ', trainFA.shape)
    # to padding for input
    margin1 = int((dFA[0] - dSeg[0]) / 2)
    margin2 = int((dFA[1] - dSeg[1]) / 2)
    margin3 = int((dFA[2] - dSeg[2]) / 2)
    two_margin1 = dFA[0] - dSeg[0]
    two_margin2 = dFA[1] - dSeg[1]
    two_margin3 = dFA[2] - dSeg[2]
    cubicCnt = 0
    marginD = [margin1, margin2, margin3]
    print('matFA shape is ', matFA.shape)
    matFAOut = np.zeros([row + two_margin1, col + two_margin2, leng + two_margin3], dtype=np.float16)
    print('matFAOut shape is ', matFAOut.shape)
    matFAOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matFA

    matSegOut = np.zeros([row + two_margin1, col + two_margin2, leng + two_margin3], dtype=np.float16)
    matSegOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matSeg

    matMaskOut = np.zeros([row + two_margin1, col + two_margin2, leng + two_margin3], dtype=np.float16)
    matMaskOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matMask

    # for mageFA, enlarge it by padding
    if margin1 != 0:
        matFAOut[0:marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matFA[marginD[0] - 1::-1, :,
                                                                                            :]  # reverse 0:marginD[0]
        matFAOut[row + marginD[0]:matFAOut.shape[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matFA[
                                                                                                                  matFA.shape[
                                                                                                                      0] - 1:row -
                                                                                                                             marginD[
                                                                                                                                 0] - 1:-1,
                                                                                                                  :,
                                                                                                                  :]  # we'd better flip it along the 1st dimension
    if margin2 != 0:
        matFAOut[marginD[0]:row + marginD[0], 0:marginD[1], marginD[2]:leng + marginD[2]] = matFA[:, marginD[1] - 1::-1,
                                                                                            :]  # we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row + marginD[0], col + marginD[1]:matFAOut.shape[1], marginD[2]:leng + marginD[2]] = matFA[
                                                                                                                  :,
                                                                                                                  matFA.shape[
                                                                                                                      1] - 1:col -
                                                                                                                             marginD[
                                                                                                                                 1] - 1:-1,
                                                                                                                  :]  # we'd flip it along the 2nd dimension
    if margin3 != 0:
        matFAOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], 0:marginD[2]] = matFA[:, :, marginD[
                                                                                                           2] - 1::-1]  # we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], marginD[2] + leng:matFAOut.shape[2]] = matFA[
                                                                                                                  :, :,
                                                                                                                  matFA.shape[
                                                                                                                      2] - 1:leng -
                                                                                                                             marginD[
                                                                                                                                 2] - 1:-1]
        # for matseg, enlarge it by padding
    if margin1 != 0:
        matSegOut[0:marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matSeg[marginD[0] - 1::-1,
                                                                                             :,
                                                                                             :]  # reverse 0:marginD[0]
        matSegOut[row + marginD[0]:matSegOut.shape[0], marginD[1]:col + marginD[1],
        marginD[2]:leng + marginD[2]] = matSeg[matSeg.shape[0] - 1:row - marginD[0] - 1:-1, :,
                                        :]  # we'd better flip it along the 1st dimension
    if margin2 != 0:
        matSegOut[marginD[0]:row + marginD[0], 0:marginD[1], marginD[2]:leng + marginD[2]] = matSeg[:,
                                                                                             marginD[1] - 1::-1,
                                                                                             :]  # we'd flip it along the 2nd dimension
        matSegOut[marginD[0]:row + marginD[0], col + marginD[1]:matSegOut.shape[1],
        marginD[2]:leng + marginD[2]] = matSeg[:, matSeg.shape[1] - 1:col - marginD[1] - 1:-1,
                                        :]  # we'd flip it along the 2nd dimension
    if margin3 != 0:
        matSegOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], 0:marginD[2]] = matSeg[:, :, marginD[
                                                                                                             2] - 1::-1]  # we'd better flip it along the 3rd dimension
        matSegOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1],
        marginD[2] + leng:matSegOut.shape[2]] = matSeg[:, :, matSeg.shape[2] - 1:leng - marginD[2] - 1:-1]

    # for matseg, enlarge it by padding
    if margin1 != 0:
        matMaskOut[0:marginD[0], marginD[1]:col + marginD[1], marginD[2]:leng + marginD[2]] = matMask[
                                                                                              marginD[0] - 1::-1, :,
                                                                                              :]  # reverse 0:marginD[0]
        matMaskOut[row + marginD[0]:matMaskOut.shape[0], marginD[1]:col + marginD[1],
        marginD[2]:leng + marginD[2]] = matMask[matMask.shape[0] - 1:row - marginD[0] - 1:-1, :,
                                        :]  # we'd better flip it along the 1st dimension
    if margin2 != 0:
        matMaskOut[marginD[0]:row + marginD[0], 0:marginD[1], marginD[2]:leng + marginD[2]] = matMask[:,
                                                                                              marginD[1] - 1::-1,
                                                                                              :]  # we'd flip it along the 2nd dimension
        matMaskOut[marginD[0]:row + marginD[0], col + marginD[1]:matMaskOut.shape[1],
        marginD[2]:leng + marginD[2]] = matMask[:, matMask.shape[1] - 1:col - marginD[1] - 1:-1,
                                        :]  # we'd flip it along the 2nd dimension
    if margin3 != 0:
        matMaskOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1], 0:marginD[2]] = matMask[:, :, marginD[
                                                                                                               2] - 1::-1]  # we'd better flip it along the 3rd dimension
        matMaskOut[marginD[0]:row + marginD[0], marginD[1]:col + marginD[1],
        marginD[2] + leng:matMaskOut.shape[2]] = matMask[:, :, matMask.shape[2] - 1:leng - marginD[2] - 1:-1]

    dsfactor = rate

    for i in range(1):
        for j in range(0, col - dSeg[1], step[1]):
            for k in range(0, leng - dSeg[2], step[2]):
                volMask = matMaskOut[i:i + dSeg[0], j:j + dSeg[1], k:k + dSeg[2]]
                if np.sum(volMask) < eps:
                    continue
                cubicCnt = cubicCnt + 1
                # index at scale 1
                volSeg = matSeg[i:i + dSeg[0], j:j + dSeg[1], k:k + dSeg[2]]
                volFA = matFAOut[i:i + dFA[0], j:j + dFA[1], k:k + dFA[2]]


                trainFA[cubicCnt, 0, :, :, :] = volFA  # 32*32*32

                trainSeg[cubicCnt, 0, :, :, :] = volSeg  # 24*24*24

    trainFA = trainFA[0:cubicCnt, :, :, :, :]
    trainSeg = trainSeg[0:cubicCnt, :, :, :, :]

    save_folder = ''
    if opt.split in ["train", "dev", "test"]:
        save_folder = os.path.join(opt.save_folder, opt.split)
    else:
        print("Specify correct split type!")
        raise FileNotFoundError
    with h5py.File(save_folder + 'train_%s.h5' % fileID, 'w') as f:
        f['noisy'] = trainFA
        f['clear'] = trainSeg

    with open('./train_list.txt', 'a') as f:
        f.write(save_folder + 'train_%s.h5' % fileID)
    return cubicCnt


def main():
    print(opt)

    # for input
    maxSource = 149.366742
    maxPercentSource = 7.76
    minSource = 0.00055037
    meanSource = 0.27593288
    stdSource = 0.75747500

    # for output
    maxTarget = 27279
    maxPercentTarget = 1320
    minTarget = -1023
    meanTarget = -601.1929
    stdTarget = 475.034

    # The image data must be a numpy saved file with the shape of NxWxH
    # N - data sample, W - image width, H - image height
    noisy_data = np.load(opt.noisy_data)
    clear_data = np.load(opt.clear_data)

    data_len = noisy_data.shape[0]

    for ind in range(data_len):
        try:
            print('source filename: ', ind)

            sourcenp = noisy_data[ind]
            sourcenp = sourcenp.reshape(1, sourcenp.shape[0], sourcenp.shape[1])

            targetnp = clear_data[ind]
            targetnp = targetnp.reshape(1, targetnp.shape[0], targetnp.shape[1])
        except RuntimeError:
            print("No image found!")
            continue

        maskimg = sourcenp

        mu = np.mean(sourcenp)

        if opt.how2normalize == 1:
            maxV, minV = np.percentile(sourcenp, [99, 1])
            print('maxV,', maxV, ' minV, ', minV)
            sourcenp = (sourcenp - mu) / (maxV - minV)
            print('unique value: ', np.unique(targetnp))

        # for training data in pelvicSeg
        if opt.how2normalize == 2:
            maxV, minV = np.percentile(sourcenp, [99, 1])
            print('maxV,', maxV, ' minV, ', minV)
            sourcenp = (sourcenp - mu) / (maxV - minV)
            print('unique value: ', np.unique(targetnp))

        # for training data in pelvicSegRegH5
        if opt.how2normalize == 3:
            std = np.std(sourcenp)
            sourcenp = (sourcenp - mu) / std
            print('maxV,', np.ndarray.max(sourcenp), ' minV, ', np.ndarray.min(sourcenp))

        if opt.how2normalize == 4:
            maxSource = 149.366742
            maxPercentSource = 7.76
            minSource = 0.00055037
            meanSource = 0.27593288
            stdSource = 0.75747500

            # for target
            maxTarget = 27279
            maxPercentTarget = 1320
            minTarget = -1023
            meanTarget = -601.1929
            stdTarget = 475.034
            
            matSource = (sourcenp - minSource) / (maxPercentSource - minSource)
            matTarget = (targetnp - meanTarget) / stdTarget

        if opt.how2normalize == 5:
            # for target
            maxTarget = 27279
            maxPercentTarget = 1320
            minTarget = -1023
            meanTarget = -601.1929
            stdTarget = 475.034

            print('target, max: ', np.amax(targetnp), ' target, min: ', np.amin(targetnp))

            # matSource = (sourcenp - meanSource) / (stdSource)
            matSource = sourcenp
            matTarget = (targetnp - meanTarget) / stdTarget

        if opt.how2normalize == 6:
            maxPercentSource, minPercentSource = np.percentile(sourcenp, [99.5, 0])
            maxPercentTarget, minPercentTarget = np.percentile(targetnp, [99.5, 0])
            print('maxPercentSource: ', maxPercentSource, ' minPercentSource: ', minPercentSource, ' maxPercentTarget: ', maxPercentTarget, 'minPercentTarget: ', minPercentTarget)

            matSource = (sourcenp - minPercentSource) / (maxPercentSource - minPercentSource) #input
            #output, use input's statistical (if there is big difference between input and output, you can find a simple relation between input and output and then include this relation to normalize output with input's statistical)
            matTarget = (targetnp - minPercentSource) / (maxPercentSource - minPercentSource) 

            print('maxSource: ', np.amax(matSource),  ' maxTarget: ', np.amax(matTarget))
            print('minSource: ', np.amin(matSource),  ' minTarget: ', np.amin(matTarget))

        fileID = str(ind)
        rate = 1
        cubicCnt = extractPatch4OneSubject(matSource, matTarget, maskimg, fileID, dSeg, step, rate)
        print('# of patches is ', cubicCnt)

        # reverse along the 1st dimension
        rmatSource = matSource[matSource.shape[0] - 1::-1, :, :]
        rmatTarget = matTarget[matTarget.shape[0] - 1::-1, :, :]

        rmaskimg = maskimg[maskimg.shape[0] - 1::-1, :, :]
        fileID = str(ind) + 'r'
        cubicCnt = extractPatch4OneSubject(rmatSource, rmatTarget, rmaskimg, fileID, dSeg, step, rate)
        print('# of patches is ', cubicCnt)


if __name__ == '__main__':
    main()
