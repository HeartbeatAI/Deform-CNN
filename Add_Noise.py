import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from utils import *
import Deform_CNN.arch as archs
from loaddataset_AddNoise import ECG


import random
import os
import argparse
import csv
import glob


arch_names = archs.__dict__.keys()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ScaledMNISTNet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: Dcnv2EcgNet)')
    parser.add_argument('--deform', default=True, type=str2bool,
                        help='use deform conv')
    parser.add_argument('--modulation', default=True, type=str2bool,
                        help='use modulated deform conv')
    parser.add_argument('--min-deform-layer', default=3, type=int,
                        help='minimum number of layer using deform conv')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.5, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--dcn', default=3, type=int,
                        help='number of layer using deform conv')
    parser.add_argument('--cvn', default=3, type=int,
                        help='number of layer using conv')
    parser.add_argument('--N', default='Null', type=str,
                        help='Add which Noise to the dataset')
    args = parser.parse_args()
    return args



    # create model

def validate(args, val_loader, model, criterion):
    

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target.long())

            acc = accuracy(output, target)[0]

            losses.update(loss.item(), input.size(0))
            scores.update(acc.item(), input.size(0))


    return output , scores.avg
def Add_Noise():
    args = parse_args()
    num_classes = 9
    model = archs.__dict__[args.arch](args, num_classes)
    model.load_state_dict(torch.load('model/ECG_Net_wDCNv2_dcn-'+str(args.dcn)+'-cvn-'+str(args.cvn)+'/model.pth'))


    print(model)
    model.cuda()
    model.eval()

    # ecg = scipy.io.loadmat(record_path)
    ###########################INFERENCE PART################################

    ## Please process the ecg data, and output the classification result.
    ## result should be an integer number in [1, 9].
    with open('./AddNOISE/DCN_'+str(args.dcn)+str(args.cvn)+'_result.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # column name
        writer.writerow(['Noise', 'Acc'])
        Noise_type=["Null","GP","JX","JD"]
        for i in Noise_type :
            print(i)
            Noise_type_name=i
            test_set = ECG(
                    train=False,
                    Noise=i
                    )
            test_loader = torch.utils.data.DataLoader(
                    
                    test_set,
                    batch_size=1,
                    #是否打乱测试数据True为是，False为否即不打乱
                    shuffle=True,
                    num_workers=8)
            
            scores = AverageMeter()
            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
   
                    input = input.cuda()
                    target = target.cuda()

                    predict= model(input)
                    

                    acc = accuracy(predict, target)[0]

               
                    scores.update(acc.item(), input.size(0))

                    _, predicted= torch.max(predict, 1)
                    result=predicted[0]



                    record_name = target

                answer = [Noise_type_name, scores.avg]
                # write result
                writer.writerow(answer)
                print(scores.avg)

        csvfile.close()
        



if __name__ == '__main__':


    result = Add_Noise()
