import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os, sys, shutil, copy, time
from torch.utils.data import Dataset, DataLoader

class YfccDataset(Dataset):
    def __init__(self, train=True, short=True):
        super(YfccDataset, self).__init__()
        
        self.data_files = []
        data_dir = '/efs/yfcc/processed/'
        if train:
            folders = [str(i) for i in range(0, 80)]
        else:
            folders = [str(i) for i in range(80, 100)]
        if short:
            folders = folders[:2]
            
        for folder in folders:
            files = os.listdir(os.path.join(data_dir, folder))
            files = [os.path.join(data_dir, folder, file) for file in files if file[-2:] == 'pt']
            self.data_files.extend(files)

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        return data[0], data[1]