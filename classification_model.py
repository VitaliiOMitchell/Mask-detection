import os
import cv2 as cv
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
import torchvision.transforms as TF
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class My_Dataset(Dataset):
    def __init__(self, path, dataset, transform=None):
        self.image_path = path
        self.dataset = pd.read_csv(dataset)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x_train, x_val, y_train, y_val = train_test_split(self.dataset.loc[:, 'image'], self.dataset.loc[:, 'label'], test_size=0.2, random_state=121)
        xmin = self.dataset.loc[index, 'xmin']
        ymin = self.dataset.loc[index, 'ymin']
        xmax = self.dataset.loc[index, 'xmax']
        ymax = self.dataset.loc[index, 'ymax']
        data_x = self.dataset.loc[index, 'image']
        data_x = cv.imread(os.path.join(self.image_path, data_x))
        data_x = cv.cvtColor(data_x, cv.COLOR_BGR2GRAY)
        data_x = data_x[ymin:ymax, xmin:xmax]
        data_x = cv.resize(data_x, (60,60))
        data_y = self.dataset.loc[index, 'labels']
        if self.transform:
            data_x = self.transform(data_x)
        
        return data_x, data_y

def make_datasets(file):
    trans = TF.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation((15)),
        transforms.ToTensor()
    ])
    

if __name__ == '__main__':
    data = My_Dataset('F:/Python/Projects/Mask-detection/images_masks', 'df_for_torch.csv')
    #print(data.__getitem__(5)[0].shape)
    plt.imshow(data.__getitem__(19)[0], cmap='gray')
    plt.show()
    

