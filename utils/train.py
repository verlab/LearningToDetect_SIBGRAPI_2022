import os, sys
from socket import socket
from random import randint, random
import torch.optim as optim
import torch.nn.functional as F
import torch, kornia
from torchvision import transforms
import torchvision
import numpy as np
import cv2, math
import glob
import tqdm
import matplotlib.pyplot as plt
import losses
import torchvision.transforms as T

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/train_gt_aug_asl_asl_ablation_matches')

# to do: path to folder
path = '/code/simulation_dataset/validation_temp/' # on proc3 only

# train
def train(net, dataset, nepochs=100, lr=1e-4, resume = None):
    ## resuming training:
    #resume = "models_train_4/5000"

    #global logdir, writer,  
    save = "models_train"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()) , lr=lr, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.9) 
    loss_vals = []

    if resume is not None:
        print('Resuming training from state...')
        net.load_state_dict(torch.load(resume + '.pth'))
        opt.load_state_dict(torch.load(resume + '-optim.pth'))
        scheduler.load_state_dict(torch.load(resume + '-scheduler.pth'))

    net.train()

    chunk_size = dataset.chunk_size if hasattr(dataset, 'chunk_size') else 1
    passes = dataset.passes if hasattr(dataset, 'passes') else 1
    epoch=0
    val_loss = torch.Tensor([]).to(device)
    train_loss_mean = []

    for subepoch in range(nepochs * chunk_size * passes):
        epoch = subepoch // (chunk_size * passes)
        step_print = (chunk_size * passes)//4
        net.train()
        train_loss = 0.
        cnt = 0
        print(scheduler.get_last_lr())
        
        val_loss = torch.Tensor([]).to(device)
        c_accumulated = 0

        with tqdm.tqdm(total=len(dataset)) as pbar:
            for batch in dataset:
                Y_img, H, W, img, Y_imgB, imgB = batch
                
                opt.zero_grad()
                loss = torch.Tensor([]).to(device)
                
                for i in range(len(Y_img)):
                    c_accumulated += 1
                    Y = Y_img[i]
                    h, w = H[i], W[i]
                    im = img[i]
                    
                    Y_B = Y_imgB[i]
                    imB = imgB[i]

                    y_map = Y.cpu().detach().numpy() 
                    y_mapB = Y_B.cpu().detach().numpy()

                    y_map = cv2.dilate(y_map, None, iterations=1)
                    y_map = cv2.GaussianBlur(y_map, (3, 3), sigmaX = 1.5)
                    y_map = (y_map-y_map.min())/(y_map.max()-y_map.min()) 
                    y_map = torch.Tensor(y_map) 

                    y_mapB = cv2.dilate(y_mapB, None, iterations=1)
                    y_mapB = cv2.GaussianBlur(y_mapB, (3, 3), sigmaX = 1.5)
                    y_mapB = (y_mapB-y_mapB.min())/(y_mapB.max()-y_mapB.min())
                    y_mapB = torch.Tensor(y_mapB) 

                    #choose random negatives
                    positives = y_map > 0
                    random_neg_idx = torch.randperm((~positives).sum())[:positives.sum()]
                    
                    positivesB = y_mapB > 0
                    random_neg_idxB = torch.randperm((~positivesB).sum())[:positivesB.sum()]

                    imT = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    imT = torch.Tensor(imT).unsqueeze(0).unsqueeze(0)
                    
                    imBT = cv2.cvtColor(imB, cv2.COLOR_BGR2GRAY)
                    imBT = torch.Tensor(imBT).unsqueeze(0).unsqueeze(0)

                    score_map = net(imT.to(device)).squeeze(0).squeeze(0)
                    score_mapB = net(imBT.to(device)).squeeze(0).squeeze(0)

                    loss_3 = losses.peak_loss2(score_map, score_mapB, y_map.to(device), y_mapB.to(device), 5).unsqueeze(0)
                    
                    score_map = torch.cat([score_map[positives], score_map[~positives][random_neg_idx]])
                    y_map = torch.cat([y_map[positives], y_map[~positives][random_neg_idx]])    
                    
                    score_mapB = torch.cat([score_mapB[positivesB], score_mapB[~positivesB][random_neg_idxB]])
                    y_mapB = torch.cat([y_mapB[positivesB], y_mapB[~positivesB][random_neg_idxB]])

                    loss_1 = losses.L2_loss2(score_map, y_map.to(device), score_mapB, y_mapB.to(device)).unsqueeze(0)
                    loss_2 = losses.cossim2(score_map, y_map.to(device), score_mapB, y_mapB.to(device)).unsqueeze(0)
                    loss_all = loss_1 + loss_2 * 3 + loss_3*0.2

                    if c_accumulated == 1:
                        val_loss = torch.cat((val_loss, loss_all), 0 )
                    else:
                        loss = torch.cat( (loss,  loss_all), 0 )
            
                loss = loss.mean()

                loss.backward()
        
                opt.step()  
                train_loss += loss.detach().item()
                cnt+=1

                train_loss_mean.append(train_loss/cnt)

                pbar.set_description('(Epc {:d}|{:d} - #step {:d} ) - Loss: {:.4f} '.format(epoch, nepochs, subepoch, train_loss / cnt))
                pbar.update(1)

            
            if subepoch % step_print == 0:   
                print('subepoch: ', subepoch)
                # ...log the running loss
                writer.add_scalar('Training loss', np.asarray(train_loss_mean).mean(), subepoch)
                val_loss_mean = val_loss.mean().detach().item()
                print('validation loss:', val_loss)
                writer.add_scalar('Validation loss', val_loss_mean, subepoch)
                train_loss_mean = []

        if save is not None and subepoch%100 == 0:
            torch.save(net.state_dict(), os.path.join(save, '{:d}.pth'.format(subepoch)))
            torch.save(opt.state_dict(), os.path.join(save, '{:d}-optim.pth'.format(subepoch)))
            torch.save(scheduler.state_dict(), os.path.join(save, '{:d}-scheduler.pth'.format(subepoch)))
            
        scheduler.step()       

    # Plot rectified patches after training
    net.eval()