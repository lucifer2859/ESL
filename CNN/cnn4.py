import os
import cv2
import numpy as np
import random
import sys

import torch
from torch import nn
from torch import optim
import torch.utils.data

# public leaderboard
# 130 | 1500 : 0.77392

epoch_num = 100000
batch_size = 64
lr = 0.001

train_path = '/home/dchen/dataset/ESL/quickdraw-data/train/'
test_path = '/home/dchen/dataset/ESL/quickdraw-data/released_test/'

model_path = '../models/cnn4.pkl'

category_mapping = {'airplane': 0, 'ant': 1, 'bear': 2, 'bird': 3, 'bridge': 4,
     'bus'     : 5, 'calendar': 6, 'car': 7, 'chair': 8, 'dog': 9,
     'dolphin' : 10, 'door': 11, 'flower': 12, 'fork': 13, 'truck': 14}

data_pairs =[]
 
def load_data(data_path, train=True):
    if (train):
        labels = os.listdir(data_path)
        for label in labels:    
            filepath = data_path + label
            filename  = os.listdir(filepath)
            for fname in filename:
                ffpath = filepath + "/" + fname
                data_pair = [ffpath, category_mapping[label]]
                data_pairs.append(data_pair)
    
        data_cnt = len(data_pairs)
        data_x = np.empty((data_cnt, 1, 28, 28), dtype="float32")
        data_y = []

        random.shuffle(data_pairs)

        i = 0
        for data_pair in data_pairs:
            img = cv2.imread(data_pair[0], 0)
            img = cv2.resize(img, (28, 28))
            arr = np.asarray(img, dtype="float32")
            data_x[i, :, :, :] = arr
            i += 1
            data_y.append(data_pair[1])
                
        data_x = data_x / 255
        data_y = np.asarray(data_y)
        data_x = torch.from_numpy(data_x)
        data_y = torch.from_numpy(data_y)

        dataset = torch.utils.data.TensorDataset(data_x, data_y)
            
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return loader
    
    filename = os.listdir(data_path)
    
    for fname in filename:
        ffpath = data_path + fname
        data_pair = [ffpath, fname.split('.')[0]]
        data_pairs.append(data_pair)
 
    data_cnt = len(data_pairs)

    data_x = np.empty((data_cnt, 1, 28, 28), dtype="float32")
    data_y = []

    i = 0
    for data_pair in data_pairs:       
        img = cv2.imread(data_pair[0], 0)
        img = cv2.resize(img, (28, 28))
        arr = np.asarray(img, dtype="float32")
        data_x[i, :, :, :] = arr  
        data_y.append(i)
        i += 1
            
    data_x = data_x / 255
    data_y = np.asarray(data_y)
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)

    dataset = torch.utils.data.TensorDataset(data_x, data_y)
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
     
    return loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Linear(256, 15)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = CNN()
# print(model)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

if __name__ == "__main__":
    train = True
    if (len(sys.argv) == 2):
        if sys.argv[1] == 'test':
            train = False
        elif sys.argv[1] != 'train':
            print('Usage: python cnn4.py [train/test]')
            print('       Default: train')
            exit(0)
    elif (len(sys.argv) > 2):
        print('Usage: python cnn4.py [train/test]')
        print('       Default: train')
        exit(0)

    device = torch.device("cuda")
    torch.cuda.set_device(11)

    if (train):
        model.to(device)

        trainloader = load_data(train_path)

        minLoss = -1
        lastSavedEpoch = 0

        for epoch in range(epoch_num):
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if (minLoss < 0) or (loss.item() < minLoss):
                    minLoss = loss.item()
                    torch.save(model, model_path)
                    print('save models epoch: %d, i: %d' % (epoch, i))
                    lastSavedEpoch = epoch

            print('epoch %d minLoss: %.12f (%d)' % (epoch, minLoss, lastSavedEpoch))
    else:
        model = torch.load(model_path, map_location='cuda:11').cuda().eval()

        testloader = load_data(test_path, train=False)

        print('id,categories')

        cnt = 0

        for i, (inputs, inds) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = model.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.tolist()
            inds = inds.tolist()
            for ii in range(len(inds)):
                print('%s,%d' % (data_pairs[inds[ii]][1], predicted[ii]))
        