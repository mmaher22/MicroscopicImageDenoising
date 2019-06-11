import argparse, os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch
import torch.utils.data as data_utils
from Unet2d_pytorch import UNet, ResUNet, UNet_LRes, ResUNet_LRes, Discriminator
from utils import *
import time
import SimpleITK as sitk
import os
import cv2

parser = argparse.ArgumentParser(description="PyTorch InfantSeg")

parser.add_argument("--gpuID", type=int, default=0, help="how to normalize the data")
parser.add_argument("--isSegReg", action="store_true", help="is Seg and Reg?", default=False)
parser.add_argument("--whichLoss", type=int, default=1, help="which loss to use: 1. LossL1, 2. lossRTL1, 3. MSE (default)")
parser.add_argument("--whichNet", type=int, default=4, help="which loss to use: 1. UNet, 2. ResUNet, 3. UNet_LRes and 4. ResUNet_LRes (default, 3)")
parser.add_argument("--lossBase", type=int, default=1, help="The base to multiply the lossG_G, Default (1)")
parser.add_argument("--batchSize", type=int, default=12, help="training batch size")
parser.add_argument("--numOfChannel_singleSource", type=int, default=2, help="# of channels for a 2D patch for the main modality (Default, 5)")
parser.add_argument("--numOfChannel_allSource", type=int, default=2, help="# of channels for a 2D patch for all the concatenated modalities (Default, 5)")
parser.add_argument("--isResidualEnhancement", action="store_true", help="is residual learning operation enhanced?", default=False)
parser.add_argument("--isViewExpansion", action="store_true", help="is view expanded?", default=True)
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=True)
parser.add_argument("--isSpatialDropOut", action="store_true", help="is spatial dropout used?", default=False)
parser.add_argument("--isFocalLoss", action="store_true", help="is focal loss used?", default=False)
parser.add_argument("--isSampleImportanceFromAd", action="store_true", help="is sample importance from adversarial network used?", default=False)
parser.add_argument("--dropoutRate", type=float, default=0.25, help="Spatial Dropout Rate. Default=0.25")
parser.add_argument("--lambdaAD", type=float, default=0, help="loss coefficient for AD loss. Default=0")
parser.add_argument("--adImportance", type=float, default=0, help="Sample importance from AD network. Default=0")
parser.add_argument("--isFixedRegions", action="store_true", help="Is the organ regions roughly known?", default=False)
parser.add_argument("--modelPath", default="model/exp3_60000.pt", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="testResult/sines/gen_wave_", type=str, help="prefix of the to-be-saved predicted filename")
parser.add_argument("--how2normalize", type=int, default=6, help="how to normalize the data")
parser.add_argument("--resType", type=int, default=1, help="resType: 0: segmentation map (integer); 1: regression map (continuous); 2: segmentation map + probability map")
parser.add_argument("--test_input_file_name",default='noisy_test_sines.npy',type=str, help="the input file name for testing subject")
parser.add_argument("--test_label_file_name",default='test_sines.npy',type=str, help="the label file name for testing subject")
parser.add_argument("--path_test", default="testImages/", type=str, help="path to the test dataset")
parser.add_argument("--output_file_name", default="testResult/gen_sines_with_train_adv.npy", type=str, help="path to the output file")

global opt
opt = parser.parse_args()


def main():
    print(opt)

    path_test = opt.path_test
    
    if opt.whichNet==1:
        netG = UNet(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==2:
        netG = ResUNet(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==3:
        netG = UNet_LRes(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==4:
        netG = ResUNet_LRes(in_channel=opt.numOfChannel_allSource, n_classes=1)

    netG.cuda()

    checkpoint = torch.load(opt.modelPath)
    netG.load_state_dict(checkpoint['model'])

    clear_sines = np.load(os.path.join(path_test, opt.test_label_file_name))
    noisy_sines = np.load(os.path.join(path_test, opt.test_input_file_name))
    gen_sines = []

    assert clear_sines.shape[0] == noisy_sines.shape[0]

    for ind in range(clear_sines.shape[0]):
        noisy_np = noisy_sines[ind]
        noisy_np = noisy_np.reshape(1, noisy_np.shape[0], noisy_np.shape[1])

        clear_np = clear_sines[ind]
        clear_np = clear_np.reshape(1, clear_np.shape[0], clear_np.shape[1])

        hpetnp = clear_np

        ##### specific normalization #####
        mu = np.mean(noisy_np)

        # for training data in pelvicSeg
        if opt.how2normalize == 1:
            maxV, minV = np.percentile(noisy_np, [99, 1])
            print('maxV,', maxV, ' minV, ', minV)
            noisy_np = (noisy_np - mu) / (maxV - minV)
            print('unique value: ', np.unique(clear_np))

        # for training data in pelvicSeg
        if opt.how2normalize == 2:
            maxV, minV = np.percentile(noisy_np, [99, 1])
            print('maxV,', maxV, ' minV, ', minV)
            noisy_np = (noisy_np - mu) / (maxV - minV)
            print('unique value: ', np.unique(clear_np))

        # for training data in pelvicSegRegH5
        if opt.how2normalize == 3:
            std = np.std(noisy_np)
            noisy_np = (noisy_np - mu) / std
            print('maxV,', np.ndarray.max(noisy_np), ' minV, ', np.ndarray.min(noisy_np))

        if opt.how2normalize == 4:
            maxLPET = 149.366742
            maxPercentLPET = 7.76
            minLPET = 0.00055037
            meanLPET = 0.27593288
            stdLPET = 0.75747500

            # for rsCT
            maxCT = 27279
            maxPercentCT = 1320
            minCT = -1023
            meanCT = -601.1929
            stdCT = 475.034

            # for s-pet
            maxSPET = 156.675962
            maxPercentSPET = 7.79
            minSPET = 0.00055037
            meanSPET = 0.284224789
            stdSPET = 0.7642257

            matLPET = (noisy_np - minLPET) / (maxPercentLPET - minLPET)
            matCT = (clear_np - meanCT) / stdCT
            matSPET = (hpetnp - minSPET) / (maxPercentSPET - minSPET)

        if opt.how2normalize == 5:
            # for rsCT
            maxCT = 27279
            maxPercentCT = 1320
            minCT = -1023
            meanCT = -601.1929
            stdCT = 475.034

            print('ct, max: ', np.amax(clear_np), ' ct, min: ', np.amin(clear_np))

            matLPET = noisy_np
            matCT = (clear_np - meanCT) / stdCT
            matSPET = hpetnp

        if opt.how2normalize == 6:
            maxPercentPET, minPercentPET = np.percentile(noisy_np, [99.5, 0])
            maxPercentCT, minPercentCT = np.percentile(clear_np, [99.5, 0])
            print('maxPercentPET: ', maxPercentPET, ' minPercentPET: ', minPercentPET, ' maxPercentCT: ', maxPercentCT, 'minPercentCT: ', minPercentCT)

            matLPET = (noisy_np - minPercentPET) / (maxPercentPET - minPercentPET)
            matCT = (clear_np - minPercentCT) / (maxPercentCT - minPercentCT)

        matFA = matLPET
        matGT = hpetnp

        print('matFA shape: ', matFA.shape, ' matGT shape: ', matGT.shape,' max(matFA): ',np.amax(matFA),' min(matFA): ',np.amin(matFA))
        if opt.whichNet == 3 or opt.whichNet == 4:
            matOut = testOneSubject_aver_res(matFA, matGT, [2, 64, 64], [1, 64, 64], [1, 8, 8], netG,
                                                opt.modelPath)
        else:
            matOut = testOneSubject_aver(matFA, matGT, [2, 64, 64], [1, 64, 64], [1, 8, 8], netG,
                                            opt.modelPath)
        print('matOut shape: ', matOut.shape, ' max(matOut): ',np.amax(matOut),' min(matOut): ',np.amin(matOut))
        if opt.how2normalize == 6:
            clear_estimated = matOut * (maxPercentCT - minPercentCT) + minPercentCT
        else:
            clear_estimated = matOut
        itspsnr = psnr(clear_estimated, matGT)
        clear_estimated = clear_estimated.reshape(clear_estimated.shape[1], clear_estimated.shape[2])
        print(clear_estimated)

        print('pred: ', clear_estimated.dtype, ' shape: ', clear_estimated.shape)
        print('gt: ', clear_np.dtype, ' shape: ', matGT.shape)
        print('psnr = ', itspsnr)
        volout = sitk.GetImageFromArray(clear_estimated)
        gen_sines.append(clear_estimated)
        volout = sitk.Cast(sitk.RescaleIntensity(volout, outputMinimum=0, outputMaximum=65535), sitk.sitkUInt16)
        sitk.WriteImage(volout, opt.prefixPredictedFN + '{}'.format(ind) + '.tiff')

    gen_sines_npy = np.stack(gen_sines, axis=0)
    np.save(opt.output_file_name, gen_sines_npy)

if __name__ == '__main__':
#     testGradients()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuID)
    main()
