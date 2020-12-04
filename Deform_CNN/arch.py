# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

from deform_conv_v2 import *


class DeformCNN(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d((1, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        features = []
        inplanes = 1
        outplanes = 32

        for y in range(args.dcn): 
            features.append(nn.Conv2d(inplanes, outplanes, (1,3), padding=(0,1), bias=False))
            features.append(self.relu)
            features.append(self.pool)
            features.append(nn.BatchNorm2d(outplanes))
            for i in range(args.cvn-1):
                features.append(nn.Conv2d(outplanes, outplanes, (1,3), padding=(0,1), bias=False))
                features.append(self.relu)
                features.append(self.pool)
                features.append(nn.BatchNorm2d(outplanes))
               
            features.append(DeformConv2d(outplanes, outplanes, 3, padding=1, bias=False, modulation=args.modulation))
            features.append(self.relu) 
            features.append(nn.BatchNorm2d(outplanes))
            if y !=args.dcn-1:
                inplanes=outplanes
                outplanes *=2
        

       
        
        #outplanes=outplanes/2
        self.features = nn.Sequential(*features)

        self.fc1 = nn.Linear(outplanes , outplanes )
        self.fc0 = nn.Linear(outplanes , 9)

    def forward(self, input):
        #self.rfpad=nn.ReflectionPad2d((0,0,54,54))
        #self.rfpad=nn.ReplicationPad2d((0,0,54,54))
        #x=self.rfpad(input)
        x = self.features(input)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        #x = self.fc1(x)
       # x = self.relu(x)
        output = self.fc0(x)

        return output
