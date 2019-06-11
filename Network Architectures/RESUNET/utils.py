import numpy as np
import os
import SimpleITK as sitk
import h5py
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import torch.nn.init
from torch.autograd import Variable
import ast
import argparse
import copy

#Dong add keys here
def Generator_2D_slices(path_patients,batchsize,inputKey='dataMR',outputKey='dataCT'):
    print(path_patients)
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    while True:
        
        for idx,namepatient in enumerate(patients):
            print(namepatient)            
            f=h5py.File(os.path.join(path_patients,namepatient),'r')
            dataMRptr=f[inputKey]
            dataMR=dataMRptr.value
            
            dataCTptr=f[outputKey]
            dataCT=dataCTptr.value

            dataMR=np.squeeze(dataMR)
            dataCT=np.squeeze(dataCT)
            
            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            dataCT=dataCT[idx_rnd,...]
            modulo=np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add=batchsize-modulo
                inds_toadd=np.random.randint(0,dataMR.shape[0],to_add)
                X=np.zeros((dataMR.shape[0]+to_add,dataMR.shape[1],dataMR.shape[2],dataMR.shape[3]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
                
                y=np.zeros((dataCT.shape[0]+to_add,dataCT.shape[1],dataCT.shape[2]))#dataCT
                y[:dataCT.shape[0],...]=dataCT
                y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
            else:
                X=np.copy(dataMR)                
                y=np.copy(dataCT)
  
            X=X.astype(np.float32)
            y=np.expand_dims(y, axis=3)#B,H,W,C
            y=y.astype(np.float32)
            
            #shuffle the data, by dong
            inds = np.arange(X.shape[0])
            np.random.shuffle(inds)
            X=X[inds,...]
            y=y[inds,...]
            
            print('y shape ', y.shape)                   
            for i_batch in range(int(X.shape[0]/batchsize)):
                yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])


# Dong add keys here
# We extract items from one whole Epoch: traverse the data thoroughly
def Generator_2D_slices_OneEpoch(path_patients, batchsize, inputKey='dataMR', outputKey='dataCT'):
    print(path_patients)
    patients = os.listdir(path_patients)  # every file  is a hdf5 patient
    # while True:
    #
    for idx, namepatient in enumerate(patients):
        print(namepatient)
        f = h5py.File(os.path.join(path_patients, namepatient), 'r')
        dataMRptr = f[inputKey]
        dataMR = dataMRptr.value

        dataCTptr = f[outputKey]
        dataCT = dataCTptr.value

        dataMR = np.squeeze(dataMR)
        dataCT = np.squeeze(dataCT)

        shapedata = dataMR.shape
        # Shuffle data
        idx_rnd = np.random.choice(shapedata[0], shapedata[0], replace=False)
        dataMR = dataMR[idx_rnd, ...]
        dataCT = dataCT[idx_rnd, ...]
        modulo = np.mod(shapedata[0], batchsize)
        ################## always the number of samples will be a multiple of batchsz##########################3
        if modulo != 0:
            to_add = batchsize - modulo
            inds_toadd = np.random.randint(0, dataMR.shape[0], to_add)
            X = np.zeros((dataMR.shape[0] + to_add, dataMR.shape[1], dataMR.shape[2], dataMR.shape[3]))  # dataMR
            X[:dataMR.shape[0], ...] = dataMR
            X[dataMR.shape[0]:, ...] = dataMR[inds_toadd]

            y = np.zeros((dataCT.shape[0] + to_add, dataCT.shape[1], dataCT.shape[2]))  # dataCT
            y[:dataCT.shape[0], ...] = dataCT
            y[dataCT.shape[0]:, ...] = dataCT[inds_toadd]

        else:
            X = np.copy(dataMR)
            y = np.copy(dataCT)

        X = X.astype(np.float32)
        y = np.expand_dims(y, axis=3)  # B,H,W,C
        y = y.astype(np.float32)

        # shuffle the data, by dong
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        X = X[inds, ...]
        y = y[inds, ...]

        print('y shape ', y.shape)
        for i_batch in range(int(X.shape[0] / batchsize)):
            yield (X[i_batch * batchsize:(i_batch + 1) * batchsize, ...],
                   y[i_batch * batchsize:(i_batch + 1) * batchsize, ...])


def weights_init(m):
    '''Custom weights initialization called on netG and netD.'''

    xavier=torch.nn.init.xavier_uniform_
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear')!=-1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


def dice(im1, im2,tid):
    '''This function is used to compute the dice ratio.

    Parameters:
    im1: gt
    im2 pred
    tid: the id for consideration

    Returns:
    dcs

    '''

    im1=im1==tid #make it boolean
    im2=im2==tid #make it boolean
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc


def psnr(ct_generated,ct_GT):
    '''This function is used to compute PSNR.

    Parameters:
    ct_generated and ct_GT

    Returns:
    PSNR

    '''

    print(ct_generated.shape)
    print(ct_GT.shape)

    mse=np.sqrt(np.mean((ct_generated-ct_GT)**2))
    print('mse ',mse)
    max_I=np.max([np.max(ct_generated),np.max(ct_GT)])
    print('max_I ',max_I)
    return 20.0*np.log10(max_I/mse)


def transfer_weights(model_from, model_to):
    '''For finetune or sth else to transfer the weights from other models.'''

    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys():
        if not k in wf:
            wf[k] = wt[k]
    model_to.load_state_dict(wf)


def evaluate(patch_MR, netG, modelPath):
    '''Evaluate one patch using the latest pytorch model.

    Parameters:
    patch_MR: a np array of shape [H,W,nchans]

    Returns:
    patch_CT_pred: segmentation maps for the corresponding input patch

    '''
    
    patch_MR = torch.from_numpy(patch_MR)

    patch_MR = patch_MR.unsqueeze(0)
    patch_MR = Variable(patch_MR).float().cuda()
    netG.cuda()
    netG.eval()
    res = netG(patch_MR)
    
    if isinstance(res, tuple):
        res = res[0]
    _, tmp = res.squeeze(0).max(0)
    patchOut = tmp.data.cpu().numpy().squeeze()

    return patchOut


def evaluate_reg(patch_MR, netG, modelPath):
    '''Evaluate one patch using the latest pytorch model.

    Parameters:
    patch_MR: a np array of shape [H,W,nchans]

    Returns:
    patch_CT_pred: segmentation maps for the corresponding input patch

    '''

    patch_MR = torch.from_numpy(patch_MR)

    patch_MR = patch_MR.unsqueeze(0)
    patch_MR = Variable(patch_MR).float().cuda()
    netG.cuda()
    netG.eval()
    res = netG(patch_MR)

    if isinstance(res, tuple):
        res = res[0]
    tmp = res
    patchOut = tmp.data.cpu().numpy().squeeze()

    return patchOut


def evaluate_res(patch_MR, netG, modelPath, nd=2):
    '''Evaluate one patch with long-term skip connection using the latest pytorch model.

    Parameters:
    patch_MR: a np array of shape [H,W,nchans]

    Returns:    
    patch_CT_pred: prediction maps for the corresponding input patch 

    '''

    patch_MR = torch.from_numpy(patch_MR)

    patch_MR = patch_MR.unsqueeze(0)

    shape = patch_MR.shape
    chn = shape[1]
    if nd!=2:
        patch_MR = patch_MR.unsqueeze(0)
        res_MR = patch_MR
    else:
        mid_slice = chn//2
        res_MR = patch_MR[:,mid_slice,...]

    patch_MR = Variable(patch_MR.float().cuda())
    res_MR = Variable(res_MR.float().cuda())

    netG.cuda()
    netG.eval()
    res = netG(patch_MR,res_MR)

    if isinstance(res, tuple):
        res = res[0]
    tmp = res
    patchOut = tmp.data.cpu().numpy().squeeze()

    return patchOut


def testOneSubject_aver_res(MR_image,CT_GT,MR_patch_sz,CT_patch_sz,step, netG, modelPath, nd=2):
    '''Receives an MR image and returns an segmentation label maps with the same size.

    Parameters:
    MR_image: the raw input data (after preprocessing)
    CT_GT: the ground truth data
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)

    Returns:
    matOut: the predicted segmentation map

    '''

    matFA = MR_image
    matSeg = CT_GT
    dFA = MR_patch_sz
    dSeg = CT_patch_sz

    eps = 1e-5
    [row,col,leng] = matFA.shape
    margin1 = int((dFA[0]-dSeg[0])/2)
    margin2 = int((dFA[1]-dSeg[1])/2)
    margin3 = int((dFA[2]-dSeg[2])/2)
    two_margin1 = dFA[0] - dSeg[0]
    two_margin2 = dFA[1] - dSeg[1]
    two_margin3 = dFA[2] - dSeg[2]
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    print('matFA shape is ',matFA.shape)
    print('dSeg:', dSeg)
    matFAOut = np.zeros([row + two_margin1, col + two_margin2, leng + two_margin3])
    print('matFAOut shape is ',matFAOut.shape)
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA

    if margin1 != 0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[0:marginD[0],:,:] #we'd better flip it along the first dimension
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[row-marginD[0]:matFA.shape[0],:,:] #we'd better flip it along the 1st dimension

    if margin2 != 0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]] = matFA[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]] = matFA[:,col-marginD[1]:matFA.shape[1],:] #we'd better to flip it along the 2nd dimension

    if margin3 != 0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]] = matFA[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]] = matFA[:,:,leng-marginD[2]:matFA.shape[2]]


    matOut = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))
    print('matOut shape:', matOut.shape)
    used = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))+eps
    print('last i ',row-dSeg[0])
    for i in range(1):
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                volFA = matFAOut[i:i + dFA[0], j:j + dFA[1], k:k + dFA[2]]
                temppremat = evaluate_res(volFA, netG, modelPath, nd=nd)
                if len(temppremat.shape)==2:
                    temppremat = np.expand_dims(temppremat,axis=0)
                matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+temppremat;
                used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+1;
    matOut = matOut/used
    return matOut


def testOneSubject_aver(MR_image,CT_GT,MR_patch_sz,CT_patch_sz,step, netG, modelPath, type='reg'):
    '''Receives an MR image and returns an segmentation label maps with the same size.

    We use averaging at the overlapping regions.

    Parameters:
    MR_image: the raw input data (after preprocessing)
    CT_GT: the ground truth data
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)

    Returns:
    matOut: the predicted segmentation/regression map

    '''

    matFA = MR_image
    matSeg = CT_GT
    dFA = MR_patch_sz
    dSeg = CT_patch_sz
    
    eps = 1e-5
    [row,col,leng] = matFA.shape
    margin1 = int((dFA[0]-dSeg[0])/2)
    margin2 = int((dFA[1]-dSeg[1])/2)
    margin3 = int((dFA[2]-dSeg[2])/2)
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    print('matFA shape is ',matFA.shape)
    matFAOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print('matFAOut shape is ',matFAOut.shape)
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA
    
    matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[0:marginD[0],:,:] #we'd better flip it along the first dimension
    matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA[row-marginD[0]:matFA.shape[0],:,:] #we'd better flip it along the 1st dimension
    
    matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]] = matFA[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
    matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]] = matFA[:,col-marginD[1]:matFA.shape[1],:] #we'd better to flip it along the 2nd dimension
    
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]] = matFA[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]] = matFA[:,:,leng-marginD[2]:matFA.shape[2]]
    
    
    matOut = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))
    used = np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))+eps
    print('last i ',row-dSeg[0])
    for i in range(0,row-dSeg[0]+1,step[0]):
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                volFA = matFAOut[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                if type == 'reg':
                    temppremat = evaluate_reg(volFA, netG, modelPath)
                else:
                    temppremat = evaluate(volFA, netG, modelPath)
                if len(temppremat.shape)==2:
                    temppremat = np.expand_dims(temppremat,axis=0)
                matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+temppremat;
                used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]] = used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+1;
    matOut = matOut/used
    return matOut


def testOneSubject(MR_image, CT_GT, NumOfClass, MR_patch_sz, CT_patch_sz, step, netG, modelPath):
    '''Receives an MR image and returns an segmentation label maps with the same size.

    We use majority voting at the overlapping regions.

    Parameters:
    MR_image: the raw input data (after preprocessing)
    CT_GT: the ground truth data
    NumOfClass: number of classes
    MR_patch_sz: 3x168x112?
    CT_patch_sz: 1x168x112?
    step: 1 (along the 1st dimension)
    netG: network for the generator
    modelPath: the pytorch model path (pth)

    Returns:
    matOut: the predicted segmentation map

    '''

    eps=1e-5
    
    matFA = MR_image
    matSeg = CT_GT
    dFA = MR_patch_sz
    dSeg = CT_patch_sz
    
    [row,col,leng] = matFA.shape
    margin1 = (dFA[0]-dSeg[0])/2
    margin2 = (dFA[1]-dSeg[1])/2
    margin3 = (dFA[2]-dSeg[2])/2
    cubicCnt = 0
    marginD = [margin1,margin2,margin3]
    
    matFAOut = np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = matFA

    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
    

    matFAOutScale = matFAOut
    matSegScale = matSeg
    matOut = np.zeros((matSegScale.shape[0],matSegScale.shape[1],matSegScale.shape[2],NumOfClass),dtype=np.int32)
    [row,col,leng] = matSegScale.shape
        
    cnt = 0
    for i in range(0,row-dSeg[0]+1,step[0]):
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg = matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                volFA = matFAOutScale[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                cnt = cnt + 1
                temppremat = evaluate(volFA, netG, modelPath)
                
                if len(temppremat.shape)==2:
                    temppremat = np.expand_dims(temppremat,axis=0)
                for labelInd in range(NumOfClass): #note, start from 0
                    currLabelMat = np.where(temppremat==labelInd, 1, 0) # true, vote for 1, otherwise 0
                    matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2],labelInd] = matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2],labelInd]+currLabelMat;
       
    matOut = matOut.argmax(axis=3)
    matOut = np.rint(matOut) #this is necessary to convert the data type to be accepted by NIFTI, otherwise will appear strange errors

    return matOut,matSeg
