import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class sr_Loss(nn.modules.loss._Loss):
    def __init__(self, sr_loss):
        super(sr_Loss, self).__init__()
        print('Preparing SR loss function:')

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in sr_loss.split('+'):
            weight, loss_type = loss.split('*')
            loss_function = None
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

        loss_sum = sum(losses)

        return loss_sum


class FALoss(torch.nn.Module):
    def __init__(self, subscale=0.125):
        super(FALoss, self).__init__()
        self.subscale = int(1 / subscale)

    def forward(self, feature1, feature2):
        feature1 = torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2 = torch.nn.AvgPool2d(self.subscale)(feature2)

        feature1 = feature1 / torch.norm(feature1, p=2, dim=1, keepdim=True)
        feature2 = feature2 / torch.norm(feature2, p=2, dim=1, keepdim=True)
        m_batchsize, C, height, width = feature1.size()
        # print(feature1.size())
        feature1 = feature1.view(m_batchsize, -1, width * height)  # [N,C,W*H]
        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.bmm(feature1.permute(0, 2, 1), feature1)  # [N,W*H,W*H]
        mat1 = mat1 / (height * width) ** 2
        # print(mat1)
        m_batchsize, C, height, width = feature2.size()
        feature2 = feature2.view(m_batchsize, -1, width * height)  # [N,C,W*H]
        # L2norm=torch.norm(feature2,2,1,keepdim=True).repeat(1,C,1)
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)
        # feature2=torch.div(feature2,L2norm)
        mat2 = torch.bmm(feature2.permute(0, 2, 1), feature2)  # [N,W*H,W*H]
        mat2 = mat2 / (height * width) ** 2
        # print(mat2)
        L1norm = torch.norm(mat2 - mat1, 1)

        #return L1norm / ((height * width) ** 2)
        return  L1norm


if __name__ == '__main__':
    FA_loss = FALoss()
    feature1 = torch.randn(1, 2, 8, 8)
    feature2 = torch.randn(1, 2, 8, 8)
    # feature2 = feature1
    loss = FA_loss(feature1, feature2)
    print(loss)