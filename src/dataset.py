#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
import transform
from torch.utils.data import Dataset

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs    = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

        if 'ECSSD' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.15, 112.48, 92.86]]])
            self.std       = np.array([[[ 56.36,  53.82, 54.23]]])
        elif 'DUTS' in self.kwargs['datapath']:
            self.mean      = np.array([[[124.55, 118.90, 102.94]]])
            self.std       = np.array([[[ 56.77,  55.97,  57.50]]])
        elif 'DUT-OMRON' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.61, 121.86, 114.92]]])
            self.std       = np.array([[[ 58.10,  57.16,  61.09]]])
        elif 'PASCAL-S' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.02, 112.75, 102.48]]])
            self.std       = np.array([[[ 59.81,  58.96,  60.44]]])
        elif 'HKU-IS' in self.kwargs['datapath']:
            self.mean      = np.array([[[123.58, 121.69, 104.22]]])
            self.std       = np.array([[[ 55.40,  53.55,  55.19]]])
        elif 'SOD' in self.kwargs['datapath']:
            self.mean      = np.array([[[109.91, 112.13,  93.90]]])
            self.std       = np.array([[[ 53.29,  50.45,  48.06]]])
        else:
            raise ValueError

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                imagepath = cfg.datapath + '/image/' + line.strip() + '.jpg'
                maskpath  = cfg.datapath + '/mask/'  + line.strip() + '.png'
                self.samples.append([imagepath, maskpath])

        if cfg.mode == 'train':
            self.transform = transform.Compose( transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                transform.Resize(320, 320),
                                                transform.RandomCrop(288, 288),
                                                transform.RandomHorizontalFlip(),
                                                transform.ToTensor())
        elif cfg.mode == 'test':
            self.transform = transform.Compose( transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                transform.Resize(320, 320),
                                                transform.ToTensor())
        else:
            raise ValueError
        
        self.cfg = cfg

    def __getitem__(self, idx):
        imagepath, maskpath = self.samples[idx]
        image               = cv2.imread(imagepath).astype(np.float32)[:,:,::-1]
        mask                = cv2.imread(maskpath).astype(np.float32)[:,:,::-1]
        H, W, C             = mask.shape
        image, mask         = self.transform(image, mask)
        return image, mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='train', datapath='./data/DUTS')
    data = Data(cfg)
    for i in range(1000):
        image, mask = data[i]
        image = image.permute(1,2,0).numpy()*cfg.std + cfg.mean
        mask  = mask.numpy()
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        input()
