from torchvision import datasets, transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import cv2
import numpy as np
import random
import scipy.ndimage as ndi
from tqdm import tqdm
import os
from PIL import Image
import scipy.io as sio
import numpy as np
import torch

class ECG(Dataset):
    def __init__(self, train=True):
        self.train = train
        txt = open(r'../dataset/datalist.txt','r') 
        self.data = txt.read()  
        txt.close()  
        self.n = self.data.count('\n')
        print(self.n)
        self.name = self.data.split("\n")
        self.total =self.n-1000
        x=0
        if self.train:
            self.train_labels = np.load("../dataset/label.npy")[:self.total]
            
        else:
            self.train_labels = np.load("../dataset/label.npy")[self.total:]
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

    
        if self.train:
          
            trdata=np.load("../dataset/DataSet250HzRepeatFill/"+self.name[index])
            self.train_data = trdata
      
        else:
            numb=self.total+index
            self.train_data=np.load("../dataset/DataSet250HzRepeatFill/"+self.name[numb])
        
        img, target = torch.unsqueeze(torch.from_numpy(self.train_data), dim=0).float(), self.train_labels[index]-1
        
        return img, target

    def __len__(self):
        if self.train:
            return self.train_labels.shape[0]
  
        else:
            return self.train_labels.shape[0]
            
            
            
