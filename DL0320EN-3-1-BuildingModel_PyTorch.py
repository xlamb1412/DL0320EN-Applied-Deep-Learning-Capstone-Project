import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn


import time
#from imageio import imread
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

torch.manual_seed(0)


train_csv_file = 'data/training_labels.csv'
validation_csv_file = 'data/validation_labels.csv'

train_data_dir = 'data/training_data_pytorch/'
validation_data_dir = 'data/validation_data_pytorch/'

class Dataset(Dataset):

    def __init__(self, csv_file, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_name = pd.read_csv(csv_file)
        self.len = self.data_name.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_name = self.data_dir + self.data_name.iloc[idx, 2]
        image = Image.open(img_name)
        y = self.data_name.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        return image, y


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)])

train_dataset = Dataset(transform=composed,
                        csv_file=train_csv_file,
                        data_dir=train_data_dir)

validation_dataset = Dataset(transform=composed,
                             csv_file=validation_csv_file,
                             data_dir=validation_data_dir)

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False


model.fc = nn.Linear(512, 7)

#Train model

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=15)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=10, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad], lr=0.003)

n_epochs = 20
loss_list = []
accuracy_list = []
correct = 0
n_test = len(validation_dataset)


for epoch in range(n_epochs+1):
    loss_sublist = []
    for x,y in train_loader:
        model.train()
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss_sublist.append(loss.data.item())
        loss.backward()
        optimizer.step()
    loss_list.append(np.mean(loss_sublist))

    correct = 0
    for x_test, y_test in validation_loader:
        model.eval()
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()

    accuracy = correct/n_test
    accuracy_list.append(accuracy)

plt.title("Average Loss per Epoch vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss per Epoch")
plt.plot(loss_list)
plt.axis([-.5, 20, 0, 5])
plt.show()

plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(accuracy_list)
plt.axis([-.5, 20, 0, 1])
plt.show()