import os
import cv2 as cv
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
        data_x = data_x[ymin:ymax, xmin:xmax]
        data_y = self.data.loc[index, 'labels']
        if self.transform:
            data_x = self.transform(data_x)
        
        return data_x, data_y


class Mask_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2, 0)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 0)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        # Max pool
        self.fc1 = nn.Linear(32*14*14, 27)
        self.fc2 = nn.Linear(27, 3)

    def forward(self, input):
        output = F.relu(self.conv1(input))
        output = self.conv2(output)
        output = F.relu(self.bn1(output))
        output = self.pool(output)
        output = F.relu(self.conv3(output))
        output = self.conv4(output)
        output = F.relu(self.bn2(output))
        output = self.pool(output)
        output = output.reshape(output.shape[0], -1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)

        return output


def make_datasets(path, file_train, file_val):
    trans = TF.Compose([
    transforms.ToPILImage(),
    transforms.Resize((60, 60)),
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
    acc_train = []
    acc_val = []
    epochs_range = np.arange(epochs)
    for epoch in range(epochs):
        train_correct = 0
        val_correct = 0
        model.train()
        for i, (x_train, y_train) in enumerate(train_data):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            
            opt.zero_grad()
            train_output = model(x_train)
            preds_train = torch.argmax(train_output, dim=1)
            train_correct += torch.sum(preds_train==y_train)
            train_loss = loss_func(train_output, y_train)
            train_loss.backward()
            opt.step()
        train_losses.append(train_loss.item())
        
        with torch.no_grad():
            model.eval()
            for i, (x_val,y_val) in enumerate(val_data):
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                val_output = model(x_val)
                val_preds = torch.argmax(val_output, dim=1)
                val_correct += torch.sum(val_preds==y_val)
                val_loss = loss_func(val_output, y_val)
            val_losses.append(val_loss.item())
            if val_loss.item() <= 0.20:
                torch.save(model.state_dict(), 'F:/Python/Projects/Mask-detection/mask_detector.pth')
        
        train_acc = (train_correct / len(train_dataset)).cpu()
        acc_train.append(train_acc)
        val_acc = (val_correct / len(val_dataset)).cpu()
        acc_val.append(val_acc)
        av_train = np.mean(train_losses)
        av_val = np.mean(val_losses)
        print(f'Epoch: {epoch}\n Train loss: {train_loss}, Average Train loss: {av_train}, Train accuracy: {train_acc}')
        print(f'Validation loss: {val_loss}, Average Validation loss: {av_val}, Validation accuracy: {val_acc}')
    
        best_train_loss_index = train_losses.index(min(train_losses))
        best_val_loss_index = val_losses.index(min(val_losses))
    
    plt.figure(figsize=(14, 4))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, linewidth=2, label='Train loss')
    plt.plot(epochs_range, val_losses, linewidth=2, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epochs_range, acc_train, label='Train accuracy')
    plt.plot(epochs_range, acc_val, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    return f'Best Train loss is {min(train_losses)} at {best_train_loss_index} epoch\n Best Validation loss: {min(val_losses)} at epoch {best_val_loss_index}'


if __name__ == '__main__':
    #cuda
    device = 'cuda'

    #Model
    model = Mask_CNN()
    model = model.to(device)

    #Hyperparameters
    batch_size = 520
    epochs = 12
    lr = 0.001
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    #Data
    train_dataset, val_dataset = make_datasets('F:/Python/Projects/Mask-detection/images_masks', 'df_for_train.csv', 'df_for_val.csv')
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    #Train-validate
    torch.backends.cudnn.benchmark = True
    print(train_val(model, epochs, train_data, val_data, opt, loss_func, device))