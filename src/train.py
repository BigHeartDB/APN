#!/usr/bin/python3
#coding=utf-8

import os
import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from apex import amp
import network
import dataset

def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='../data/DUTS', savepath='../out', mode='train', batch=64, lr=0.1, momen=0.9, decay=5e-4, epoch=32)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=16)
    ## network
    net    = Network.APN(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter('../log')
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask, (H,W), name) in enumerate(loader):
            image, mask            = image.cuda().float(), mask.cuda().float()
            out2, out3, out4, out5 = net(image)

            loss2                  = F.binary_cross_entropy_with_logits(out2, mask)
            loss3                  = F.binary_cross_entropy_with_logits(out3, mask)
            loss4                  = F.binary_cross_entropy_with_logits(out4, mask)
            loss5                  = F.binary_cross_entropy_with_logits(out5, mask)
            loss                   = loss2+loss3/2+loss4/4+loss5/8
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss2':loss2.item(), 'loss3':loss3.item(), 'loss4':loss4.item(), 'loss5':loss5.item(), 'loss':loss.item()}, global_step=global_step)
            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f'
                    %(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()))

        if (epoch+1)%8 == 0 or (epoch+1)==cfg.epoch:
            if not os.path.exists(cfg.savepath):
                os.makedirs(cfg.savepath)
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))


if __name__=='__main__':
    train(dataset, network)