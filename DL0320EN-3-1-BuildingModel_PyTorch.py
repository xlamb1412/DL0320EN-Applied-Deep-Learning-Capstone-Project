import torch
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn



import matplotlib.pylab as plt
import pandas as pd
from PIL import Image
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

model_r = models.resnet18(pretrained=True)

for param in model_r.parameters():
    param.requires_grad = False


model_r.fc = nn.Linear(512, 7)

#Train model

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=15)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=10, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([parameters for parameters in model_r.parameters() if parameters.requires_grad], lr=0.003)

n_epochs = 20
loss_list_r = []
accuracy_list_r = []
correct = 0
n_test = len(validation_dataset)


for epoch in range(n_epochs+1):
    loss_sublist = []
    for x,y in train_loader:
        model_r.train()
        optimizer.zero_grad()
        z = model_r(x)
        loss = criterion(z, y)
        loss_sublist.append(loss.data.item())
        loss.backward()
        optimizer.step()
    loss_list_r.append(np.mean(loss_sublist))

    correct = 0
    for x_test, y_test in validation_loader:
        model_r.eval()
        z = model_r(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()

    accuracy = correct/n_test
    accuracy_list_r.append(accuracy)

plt.title("Average Loss per Epoch vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss per Epoch")
plt.plot(loss_list_r)
plt.axis([-.5, 20, 0, 5])
plt.show()

plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(accuracy_list_r)
plt.axis([-.5, 20, 0, 1])
plt.show()

model_d = models.densenet121(pretrained=True)

for param in model_d.parameters():
    param.requires_grad = False

model_d.fc = nn.Linear(1024, 7)
#Train model

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=15)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=10, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([parameters for parameters in model_d.parameters() if parameters.requires_grad], lr=0.003)

n_epochs = 10
loss_list_d = []
accuracy_list_d = []
correct = 0
n_test = len(validation_dataset)


for epoch in range(n_epochs+1):
    loss_sublist_d = []
    for x,y in train_loader:
        model_d.train()
        optimizer.zero_grad()
        z = model_d(x)
        loss = criterion(z, y)
        loss_sublist_d.append(loss.data.item())
        loss.backward()
        optimizer.step()
    loss_list_d.append(np.mean(loss_sublist_d))

    correct = 0
    for x_test, y_test in validation_loader:
        model_d.eval()
        z = model_d(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()

    accuracy_d = correct/n_test
    accuracy_list_d.append(accuracy_d)

    plt.title("Average Loss per Epoch vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss per Epoch")
    plt.plot(loss_list_d)
    plt.axis([-.5, 20, 0, 5])
    plt.show()

    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(accuracy_list_d)
    plt.axis([-.5, 20, 0, 1])
    plt.show()