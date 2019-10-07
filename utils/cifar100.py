#Filename:	cifar100.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 29 Jul 2019 09:56:48  +08

import pickle
import numpy as np
import os
import torch
import torch.utils.data as Data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class CIFAR(Data.Dataset):

    def __init__(self, filename, transform = None):
        self.transform = transform
        self.filename = filename
        with open(self.filename, 'rb') as f:
            data = pickle.load(f, encoding = 'bytes')

        self.length = len(data[b'data'])
        self.images = data[b'data']
        self.labels = data[b'fine_labels']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = np.reshape(image, [3, 32, 32])
        image = np.transpose(image, [1, 2, 0])
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

if __name__ == "__main__":
    filename = "/home/yongjie/code/Hierarchical_focal_loss/data/CIFAR100/cifar-100-python/train"
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((32, 32), padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                             std=[0.2673, 0.2564, 0.2762]),
        ])

    cifar100 = CIFAR(filename, train_transform)
    train = Data.DataLoader(dataset = cifar100, batch_size = 10, shuffle = True, num_workers = 20)
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(train):
            print(epoch, batch_x.shape, batch_y.shape)

