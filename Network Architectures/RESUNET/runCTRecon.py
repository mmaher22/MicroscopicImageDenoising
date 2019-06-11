import argparse, os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch
import torch.utils.data as data_utils
from utils import *
from Unet2d_pytorch import UNet, ResUNet, UNet_LRes, ResUNet_LRes, Discriminator
from nnBuildUnits import CrossEntropy3d, topK_RegLoss, RelativeThreshold_RegLoss, gdl_loss, adjust_learning_rate, calc_gradient_penalty
import time
import SimpleITK as sitk

# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--gpuID", type=int, default=0, help="how to normalize the data")
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=True)
parser.add_argument("--isWDist", action="store_true", help="is adversarial loss with WGAN-GP distance?", default=True)
parser.add_argument("--lambda_AD", default=0.05, type=float, help="weight for AD loss, Default: 0.05")
parser.add_argument("--lambda_D_WGAN_GP", default=10, type=float, help="weight for gradient penalty of WGAN-GP, Default: 10")
parser.add_argument("--how2normalize", type=int, default=6, help="how to normalize the data")
parser.add_argument("--whichLoss", type=int, default=1, help="which loss to use: 1. LossL1, 2. lossRTL1, 3. MSE (default)")
parser.add_argument("--isGDL", action="store_true", help="do we use GDL loss?", default=True)
parser.add_argument("--gdlNorm", default=2, type=int, help="p-norm for the gdl loss, Default: 2")
parser.add_argument("--lambda_gdl", default=0.05, type=float, help="Weight for gdl loss, Default: 0.05")
parser.add_argument("--whichNet", type=int, default=4, help="which loss to use: 1. UNet, 2. ResUNet, 3. UNet_LRes and 4. ResUNet_LRes (default, 3)")
parser.add_argument("--lossBase", type=int, default=1, help="The base to multiply the lossG_G, Default (1)")
parser.add_argument("--batchSize", type=int, default=12, help="training batch size")
parser.add_argument("--numOfChannel_singleSource", type=int, default=2, help="# of channels for a 2D patch for the main modality (Default, 5)")
parser.add_argument("--numOfChannel_allSource", type=int, default=2, help="# of channels for a 2D patch for all the concatenated modalities (Default, 5)")
parser.add_argument("--numofIters", type=int, default=200000, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=100, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=5000, help="number of iterations to save the model")
parser.add_argument("--showValPerformanceEvery", type=int, default=1000, help="number of iterations to show validation performance")
parser.add_argument("--showTestPerformanceEvery", type=int, default=1000, help="number of iterations to show test performance")
parser.add_argument("--lr", type=float, default=5e-3, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr_netD", type=float, default=5e-3, help="Learning Rate for discriminator. Default=5e-3")
parser.add_argument("--dropout_rate", default=0.2, type=float, help="prob to drop neurons to zero: 0.2")
parser.add_argument("--decLREvery", type=int, default=10000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument("--lrDecRate", type=float, default=0.5, help="The weight for decreasing learning rate of netG Default=0.5")
parser.add_argument("--lrDecRate_netD", type=float, default=0.1, help="The weight for decreasing learning rate of netD. Default=0.1")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default=True)
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--RT_th", default=0.005, type=float, help="Relative thresholding: 0.005")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--prefixModelName", default="model/exp4_", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="exp4_", type=str, help="prefix of the to-be-saved predicted filename")
parser.add_argument("--test_input_file_name",default='test_data_patches.npy',type=str, help="the input file name for testing subject")
parser.add_argument("--test_label_file_name",default='test_label_patches.npy',type=str, help="the label file name for testing subject")
parser.add_argument("--path_test", default="testImages/", type=str, help="path to the test dataset")
parser.add_argument("--path_train", default="data/train/", type=str, help="path to the train dataset")
parser.add_argument("--path_dev", default="data/val/", type=str, help="path to the dev dataset")

global opt, model 
opt = parser.parse_args()

def main():
    print(opt)

    netD = Discriminator()
    netD.apply(weights_init)
    netD.cuda()
    
    optimizerD = optim.Adam(netD.parameters(),lr=opt.lr_netD)
    criterion_bce=nn.BCELoss()
    criterion_bce.cuda()
    
    if opt.whichNet==1:
        net = UNet(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==2:
        net = ResUNet(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==3:
        net = UNet_LRes(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==4:
        net = ResUNet_LRes(in_channel=opt.numOfChannel_allSource, n_classes=1, dp_prob = opt.dropout_rate)
    net.cuda()
    params = list(net.parameters())
    print('len of params is ')
    print(len(params))
    print('size of params is ')
    print(params[0].size())
    
    optimizer = optim.Adam(net.parameters(),lr=opt.lr)
    criterion_L2 = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    criterion_RTL1 = RelativeThreshold_RegLoss(opt.RT_th)
    criterion_gdl = gdl_loss(opt.gdlNorm)
    
    given_weight = torch.cuda.FloatTensor([1,4,4,2])
    
    criterion_3d = CrossEntropy3d(weight=given_weight)
    
    criterion_3d = criterion_3d.cuda()
    criterion_L2 = criterion_L2.cuda()
    criterion_L1 = criterion_L1.cuda()
    criterion_RTL1 = criterion_RTL1.cuda()
    criterion_gdl = criterion_gdl.cuda()
    
    path_test = opt.path_test
    path_train = opt.path_train
    path_dev = opt.path_dev
    data_generator = Generator_2D_slices(path_train, opt.batchSize, inputKey='noisy',outputKey='clear')
    data_generator_dev = Generator_2D_slices(path_dev, opt.batchSize, inputKey='noisy', outputKey='noisy')

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            net.load_state_dict(checkpoint['model'])
            opt.start_epoch = checkpoint["epoch"] + 1
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
    running_loss = 0.0
    start = time.time()
    for iter in range(opt.start_epoch, opt.numofIters+1):
        inputs, labels = next(data_generator)
        exinputs = inputs

        inputs = np.squeeze(inputs) #5x64x64
        exinputs = np.squeeze(exinputs) #5x64x64
        labels = np.squeeze(labels) #64x64

        inputs = inputs.astype(float)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.float()
        exinputs = exinputs.astype(float)
        exinputs = torch.from_numpy(exinputs)
        exinputs = exinputs.float()
        labels = labels.astype(float)
        labels = torch.from_numpy(labels)
        labels = labels.float()
        source = inputs
        mid_slice = opt.numOfChannel_singleSource//2
        residual_source = inputs[:, mid_slice, ...]
        source = source.cuda()
        residual_source = residual_source.cuda()
        labels = labels.cuda()
        
        #wrap them into Variable
        source, residual_source, labels = Variable(source),Variable(residual_source), Variable(labels)
        
        ## (1) update D network: maximize log(D(x)) + log(1 - D(G(z)))
        if opt.isAdLoss:
            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source)  # 5x64x64->1*64x64
            else:
                outputG = net(source)  # 5x64x64->1*64x64
                
            if len(labels.size())==3:
                labels = labels.unsqueeze(1)
                
            outputD_real = netD(labels)
            outputD_real = torch.sigmoid(outputD_real)
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
                
            outputD_fake = netD(outputG)
            outputD_fake = torch.sigmoid(outputD_fake)
            netD.zero_grad()
            batch_size = inputs.size(0)
            real_label = torch.ones(batch_size,1)
            real_label = real_label.cuda()
            real_label = Variable(real_label)
            loss_real = criterion_bce(outputD_real,real_label)
            loss_real.backward()
            #train with fake data
            fake_label = torch.zeros(batch_size,1)
            fake_label = fake_label.cuda()
            fake_label = Variable(fake_label)
            loss_fake = criterion_bce(outputD_fake,fake_label)
            loss_fake.backward()
            
            lossD = loss_real + loss_fake
            #update network parameters
            optimizerD.step()
            
        if opt.isWDist:
            one = torch.FloatTensor([1])
            mone = one * -1
            one = one.cuda()
            mone = mone.cuda()
            
            netD.zero_grad()
            
            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source)  # 5x64x64->1*64x64
            else:
                outputG = net(source)  # 5x64x64->1*64x64
                
            if len(labels.size())==3:
                labels = labels.unsqueeze(1)
                
            outputD_real = netD(labels)
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
                
            outputD_fake = netD(outputG)

            
            batch_size = inputs.size(0)
            
            D_real = outputD_real.mean()
            D_real.backward(mone)
        
        
            D_fake = outputD_fake.mean()
            D_fake.backward(one)
        
            gradient_penalty = opt.lambda_D_WGAN_GP*calc_gradient_penalty(netD, labels.data, outputG.data)
            gradient_penalty.backward()
            
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            
            optimizerD.step()
        
        
        ## (2) update G network: minimize the L1/L2 loss, maximize the D(G(x))
        if opt.whichNet == 3 or opt.whichNet == 4:
            outputG = net(source, residual_source)  # 5x64x64->1*64x64
        else:
            outputG = net(source)  # 5x64x64->1*64x64
        net.zero_grad()
        if opt.whichLoss==1:
            lossG_G = criterion_L1(torch.squeeze(outputG), torch.squeeze(labels))
        elif opt.whichLoss==2:
            lossG_G = criterion_RTL1(torch.squeeze(outputG), torch.squeeze(labels))
        else:
            lossG_G = criterion_L2(torch.squeeze(outputG), torch.squeeze(labels))
        lossG_G = opt.lossBase * lossG_G
        lossG_G.backward(retain_graph=True) #compute gradients

        if opt.isGDL:
            lossG_gdl = opt.lambda_gdl * criterion_gdl(outputG,torch.unsqueeze(torch.squeeze(labels,1),1))
            lossG_gdl.backward() #compute gradients

        if opt.isAdLoss:
            #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
            #angel of equation (note the max and min difference for generator and discriminator)
            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source)  # 5x64x64->1*64x64
            else:
                outputG = net(source)  # 5x64x64->1*64x64
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            
            outputD = netD(outputG)
            outputD = torch.sigmoid(outputD)
            lossG_D = opt.lambda_AD*criterion_bce(outputD,real_label) #note, for generator, the label for outputG is real, because the G wants to confuse D
            lossG_D.backward()
            
        if opt.isWDist:
            #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
            #angel of equation (note the max and min difference for generator and discriminator)
            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source)  # 5x64x64->1*64x64
            else:
                outputG = net(source)  # 5x64x64->1*64x64
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            
            outputD_fake = netD(outputG)

            outputD_fake = outputD_fake.mean()
            
            lossG_D = opt.lambda_AD*outputD_fake.mean() #note, for generator, the label for outputG is real, because the G wants to confuse D
            lossG_D.backward(mone)
        
        #for other losses, we can define the loss function following the pytorch tutorial
        
        optimizer.step() #update network parameters

        running_loss = running_loss + lossG_G.data.item()

        
        if iter % opt.showTrainLossEvery==0: #print every 2000 mini-batches
            print('************************************************')
            print('time now is: ' + time.asctime(time.localtime(time.time())))
            print('average running loss for generator between iter [%d, %d] is: %.5f'%(iter - 100 + 1, iter, running_loss/100))
            
            print('lossG_G is %.5f respectively.' % (lossG_G.data.item()))

            if opt.isGDL:
                print('loss for GDL loss is %f' % lossG_gdl.data.item())

            if opt.isAdLoss:
                print('loss for discriminator is %f' % lossD.data.item())
                print('lossG_D for discriminator is %f' % lossG_D.data.item())

            if opt.isWDist:
                print('loss_real is ', torch.mean(D_real).data.item(), 'loss_fake is ', torch.mean(D_fake).data.item())
                print('loss for discriminator is %f' % Wasserstein_D.data.item(), ' D cost is %f' % D_cost)                
                print('lossG_D for discriminator is %f' % lossG_D.data.item())
  
            print('cost time for iter [%d, %d] is %.2f'%(iter - 100 + 1, iter, time.time()-start))
            print('************************************************')
            running_loss = 0.0
            start = time.time()
        if iter%opt.saveModelEvery==0: #save the model
            state = {
                'epoch': iter+1,
                'model': net.state_dict()
            }
            torch.save(state, opt.prefixModelName + '%d.pt' % iter)
            print('save model: '+ opt.prefixModelName + '%d.pt' % iter)

            if opt.isAdLoss or opt.isWDist:
                torch.save(netD.state_dict(), opt.prefixModelName+'_net_D%d.pt' % iter)
        if iter%opt.decLREvery==0:
            opt.lr = opt.lr * opt.lrDecRate
            adjust_learning_rate(optimizer, opt.lr)
            if opt.isAdLoss or opt.isWDist:
                opt.lr_netD = opt.lr_netD * opt.lrDecRate_netD
                adjust_learning_rate(optimizerD, opt.lr_netD)

                
        if iter%opt.showValPerformanceEvery==0: #test one subject
            # to test on the validation dataset in the format of h5 
            inputs, labels = next(data_generator_dev)
            exinputs = inputs

            inputs = np.squeeze(inputs)

            exinputs = np.squeeze(exinputs)  # 5x64x64

            labels = np.squeeze(labels)

            inputs = torch.from_numpy(inputs)
            inputs = inputs.float()
            exinputs = torch.from_numpy(exinputs)
            exinputs = exinputs.float()
            labels = torch.from_numpy(labels)
            labels = labels.float()
            mid_slice = opt.numOfChannel_singleSource // 2
            residual_source = inputs[:, mid_slice, ...]
            source = inputs
            source = source.cuda()
            residual_source = residual_source.cuda()
            labels = labels.cuda()
            source,residual_source,labels = Variable(source),Variable(residual_source), Variable(labels)

            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source)  # 5x64x64->1*64x64
            else:
                outputG = net(source)  # 5x64x64->1*64x64
            if opt.whichLoss == 1:
                lossG_G = criterion_L1(torch.squeeze(outputG), torch.squeeze(labels))
            elif opt.whichLoss == 2:
                lossG_G = criterion_RTL1(torch.squeeze(outputG), torch.squeeze(labels))
            else:
                lossG_G = criterion_L2(torch.squeeze(outputG), torch.squeeze(labels))
            lossG_G = opt.lossBase * lossG_G
            print('.......come to validation stage: iter {}'.format(iter),'........')
            print('lossG_G is %.5f.'%(lossG_G.data.item()))

            if opt.isGDL:
                lossG_gdl = criterion_gdl(outputG, torch.unsqueeze(torch.squeeze(labels,1),1))
                print('loss for GDL loss is %f'%lossG_gdl.data.item())

        if iter % opt.showTestPerformanceEvery == 0:  # test one subject
            noisy_np = np.load(os.path.join(path_test, opt.test_input_file_name))
            noisy_np = noisy_np[0]
            noisy_np = noisy_np.reshape(1, noisy_np.shape[0], noisy_np.shape[1])

            clear_np = np.load(os.path.join(path_test, opt.test_label_file_name))
            clear_np = clear_np[0]
            clear_np = clear_np.reshape(1, clear_np.shape[0], clear_np.shape[1])


            hpetnp = clear_np

            #for training data in pelvicSeg
            if opt.how2normalize == 1:
                maxV, minV = np.percentile(noisy_np, [99 ,1])
                print('maxV,',maxV,' minV, ',minV)
                noisy_np = (noisy_np-mu)/(maxV-minV)
                print('unique value: ',np.unique(clear_np))

            #for training data in pelvicSeg
            if opt.how2normalize == 2:
                maxV, minV = np.percentile(noisy_np, [99 ,1])
                print('maxV,',maxV,' minV, ',minV)
                noisy_np = (noisy_np-mu)/(maxV-minV)
                print('unique value: ',np.unique(clear_np))

            #for training data in pelvicSegRegH5
            if opt.how2normalize== 3:
                std = np.std(noisy_np)
                noisy_np = (noisy_np - mu)/std
                print('maxV,',np.ndarray.max(noisy_np),' minV, ',np.ndarray.min(noisy_np))

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
                matSPET = (hpetnp - minPercentPET) / (maxPercentPET - minPercentPET)

                matCT = (clear_np - minPercentCT) / (maxPercentCT - minPercentCT)


            matFA = matLPET
            matGT = hpetnp

            print('matFA shape: ', matFA.shape, ' matGT shape: ', matGT.shape)
            matOut = testOneSubject_aver_res(matFA, matGT, [2,64,64], [1,64,64], [1,8,8], net,opt.prefixModelName+'%d.pt'%iter)
            print('matOut shape: ', matOut.shape)
            if opt.how2normalize==6:
                clear_estimated = matOut * (maxPercentPET - minPercentPET) + minPercentPET
            else:
                clear_estimated = matOut

            itspsnr = psnr(clear_estimated, matGT)
            clear_estimated = clear_estimated.reshape(clear_estimated.shape[1], clear_estimated.shape[2])

            print('pred: ', clear_estimated.dtype, ' shape: ', clear_estimated.shape)
            print('gt: ', clear_np.dtype, ' shape: ', clear_estimated.shape)
            print('psnr = ', itspsnr)
            volout = sitk.GetImageFromArray(clear_estimated)
            volout = sitk.Cast(sitk.RescaleIntensity(volout, outputMinimum=0, outputMaximum=65535), sitk.sitkUInt16)
            sitk.WriteImage(volout,opt.prefixPredictedFN+'{}'.format(iter)+'.tiff')
            np.save(opt.prefixPredictedFN+'{}'.format(iter)+'.npy', clear_estimated)

        
    print('Finished Training')
    
if __name__ == '__main__':   
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuID)  
    main()
    
