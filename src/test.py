#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
import network

class Test(object):
    def __init__(self, Dataset, Network, Datapath):
        ## dataset
        self.cfg    = Dataset.Config(datapath=Datapath, snapshot='../out/model-32', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network.APN(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def accuracy(self):
        with torch.no_grad():
            mae, fscore, cnt, number   = 0, 0, 0, 256
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)

            for image, mask, (H, W), name in self.loader:
                image, mask            = image.cuda().float(), mask.cuda().float()
                out2, out3, out4, out5 = self.net(image)
                pred                   = torch.sigmoid(out2)

                ## MAE
                cnt += 1
                mae += (pred-mask).abs().mean()
                ## F-Score
                precision = torch.zeros(number)
                recall    = torch.zeros(number)
                for i in range(number):
                    temp         = (pred >= threshod[i]).float()
                    precision[i] = (temp*mask).sum()/(temp.sum()+1e-12)
                    recall[i]    = (temp*mask).sum()/(mask.sum()+1e-12)
                mean_pr += precision
                mean_re += recall
                fscore   = mean_pr*mean_re*(1+0.3)/(0.3*mean_pr+mean_re+1e-12)
                if cnt % 20 == 0:
                    print('MAE=%.6f, F-score=%.6f'%(mae/cnt, fscore.max()/cnt))
            print('MAE=%.6f, F-score=%.6f'%(mae/cnt, fscore.max()/cnt))

    def show(self):
        with torch.no_grad():
            for image, mask, (H, W), name in self.loader:
                image, mask            = image.cuda().float(), mask.cuda().float()
                out2, out3, out4, out5 = self.net(image)
                out = out2
                plt.subplot(221)
                plt.imshow(np.uint8(image[0].permute(1,2,0).cpu().numpy()*self.cfg.std + self.cfg.mean))
                plt.subplot(222)
                plt.imshow(mask[0, 0].cpu().numpy())
                plt.subplot(223)
                plt.imshow(out[0, 0].cpu().numpy())
                plt.subplot(224)
                plt.imshow(torch.sigmoid(out[0, 0]).cpu().numpy())
                plt.show()
                cv2.imwrite('s.jpg', np.uint8(image[0].permute(1,2,0).cpu().numpy()[:,:,::-1]*self.cfg.std + self.cfg.mean))
                input()
    
    def save(self):
        with torch.no_grad():
            for image, mask, (H, W), name in self.loader:
                image, shape           = image.cuda().float(), (H, W)
                out2, out3, out4, out5 = self.net(image, shape)
                pred                   = (torch.sigmoid(out2[0,0])*255).cpu().numpy()
                head                   = '../eval/APN/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0], np.uint8(pred))


if __name__=='__main__':
    paths = ['../data/SOD', '../data/PASCAL-S', '../data/ECSSD', '../data/DUTS', '../data/HKU-IS', '../data/DUT-OMRON']
    for path in paths:
        Test(dataset, network, path).save()
