import os
import cv2 as cv
import PIL 
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
    def __init__(self, path, data, transform=None):
        self.image_path = path
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        xmin = self.data.loc[index, 'xmin']
        ymin = self.data.loc[index, 'ymin']
        xmax = self.data.loc[index, 'xmax']
        ymax = self.data.loc[index, 'ymax']
        image = self.data.loc[index, 'image']
        data_x = cv.imread(os.path.join(self.image_path, image))
        data_x = cv.cvtColor(data_x, cv.COLOR_BGR2GRAY)
        data_x = data_x[ymin:ymax, xmin:xmax]
        data_x = cv.resize(data_x, (60,60))
        data_x = PIL.Image.fromarray(data_x)
        data_y = self.data.loc[index, 'labels']
        if self.transform:
            data_x = self.transform(data_x)
        
        return data_x, data_y, #image


class Mask_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.batch1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0) 
        # Max pool
        #self.conv4 = nn.Conv2d(128, 128, 3, 1, 1) 
        self.batch2 = nn.BatchNorm2d(128) 
        self.drop1 = nn.Dropout2d(p=0.3)
        # Max pool
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 1) #before it was 128 -> 32 
        self.fc1 = nn.Linear(64*7*7, 64) #it was 32*7*7 -> 16
        self.drop2 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(64, 32)
        self.drop3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, input):
        output = F.relu(self.conv1(input))
        #return output.shape
        output = self.pool(output)
        output = self.conv2(output)
        output = F.relu(self.batch1(output))
        output = F.relu(self.conv3(output))
        output = self.pool(output)
        #output = F.relu(self.conv4(output))
        #output = self.conv4(output)
        output = F.relu(self.batch2(output))
        output = self.drop1(output)
        output = self.pool(output)
        output = F.relu(self.conv5(output))
        output = output.reshape(output.shape[0], -1)
        output = self.fc1(output)
        output = self.drop2(output)
        output = self.fc2(output)
        output = self.drop3(output)
        output = self.fc3(output)

        return output


def make_datasets(path, file_train, file_val):
    trans = TF.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor()
    ])
    df_train = pd.read_csv(file_train)
    df_val = pd.read_csv(file_val)
    
    train_dataset = My_Dataset(path, df_train, transform=trans)
    val_dataset = My_Dataset(path, df_val, transform=trans)

    return train_dataset, val_dataset

def train_val(model, epochs, train_data, val_data, opt, loss_func, device):
    train_losses = []
    val_losses = []
    #train_acc = 0
    #val_acc = 0
    epochs_range = np.arange(epochs)
    for epoch in range(epochs):
        model.train()
        for i, (x_train, y_train) in enumerate(train_data):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            
            opt.zero_grad()
            train_output = model(x_train)
            #output_for_edit = torch.argmax(output, dim=1)
            train_loss = loss_func(train_output, y_train)
            train_loss.backward()
            opt.step()
        train_losses.append(train_loss.item())
        #train_acc += torch.sum(train_output==y_train) / len(train_dataset)
    
        #print(f'Epoch: {epoch}, loss:{}')
        with torch.no_grad():
            model.eval()
            for i, (x_val,y_val) in enumerate(val_data):
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                val_output = model(x_val)
                val_loss = loss_func(val_output, y_val)
            val_losses.append(val_loss.item())
            #val_acc += torch.sum(val_output==y_val) / len(val_dataset)
            if val_loss.item() <= 0.1:
                torch.save(model.state_dict('F:/Python/Projects/Mask-detection/mask_detector.pth'))
        
        av_train = np.mean(train_losses)
        av_val = np.mean(val_losses)
        print(f'Epoch: {epoch}\n Train loss: {train_loss}, Average Train loss:{av_train}')
        print(f'Validation loss: {val_loss}, Average Validation loss: {av_val}')
    
        best_train_loss_index = train_losses.index(min(train_losses))
        best_val_loss_index = val_losses.index(min(val_losses))
    
    #plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, linewidth=2, label='Train loss')
    plt.plot(epochs_range, val_losses, linewidth=2, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    #plt.subplot(1,2,2)
    #plt.plot(epochs_range, train_acc, label='Train accuracy')
    #plt.plot(epochs_range, val_acc, label='Validation accuracy')
    #plt.xlabel('Epochs')
    #plt.ylabel('Accuracy')
    #plt.legend()
    #plt.show()
    
    return f'Best Train loss is {min(train_losses)} at {best_train_loss_index} epoch\n Best Validation loss: {min(val_losses)} at epoch {best_val_loss_index}'

if __name__ == '__main__':
    #cuda
    device = 'cuda'

    #Model
    model = Mask_CNN()
    model = model.to(device)

    #Hyperparameters
    batch_size = 520
    epochs = 3
    lr = 0.001
    #opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    #opt = optim.SGD(model.parameters(), lr=lr)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    #Data
    train_dataset, val_dataset = make_datasets('F:/Python/Projects/Mask-detection/images_masks', 'df_for_train.csv', 'df_for_val.csv')
    train_data = DataLoader(train_dataset, batch_size=batch_size)
    val_data = DataLoader(val_dataset, batch_size=batch_size)

    #Train-validate
    torch.backends.cudnn.benchmark = True
    print(train_val(model, epochs, train_data, val_data, opt, loss_func, device))

    


