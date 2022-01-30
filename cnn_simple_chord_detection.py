
import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as tt

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

############## CONSTANTS ##############
HEIGHT, WIDTH = 124, 512




dataset = ImageFolder(data_dir+'/train', transform=ToTensor())
count_dic = {0:0, 1:0, 2:0, 3:0, 4:0}
for im, label in dataset:  
    count_dic[label] += 1
    #print('label: ', label)
print('count_dic\n', count_dic)

# Mapping from index label to its actual value
index_to_label = {}
for i, label in enumerate(classes):
    index_to_label[i] = label
index_to_label

def get_mean_std(dataset):
    for im, _ in dataset:    
        channels_sum, channels_squared_sum = 0, 0    
        channels_sum += torch.mean(im, dim=[1,2])
        channels_squared_sum += torch.mean(im**2, dim=[1,2])

    mean = channels_sum/len(dataset)
    std = (channels_squared_sum/len(dataset) - mean**2)**0.5
    return mean, std
    
# mean, std = get_mean_std(dataset)
# print("mean: {}, std: {}".format(mean, std))

mean, std = get_mean_std(dataset)
stats = [[item.item() for item in mean], [item.item() for item in std]]
stats

# Data transforms (normalization & data augmentation)
train_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats, inplace=True)])

train_ds = ImageFolder(data_dir+'/train', train_tfms)

for img, _ in train_ds:    
    img = (img-torch.min(img))/torch.max(img)
    plt.imshow(img.permute(1,2,0));plt.show()
    break
dataset = train_ds

# for img, label in dataset:
#     print(img.shape)
#     plt.imshow(img.permute(1,2,0));plt.show()
#     print(label)
#     break
    
val_size = int(0.2*len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

batch_size = 16
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        #print('training step:\nimages shape: ', images.shape, '\n labels: ', labels)
        out = self(images)                  # Generate predictions
        #print('out shape: ', out.shape)
        #print('labels shape: ', labels.shape)
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        #print('images shape: ', images)
        #print('label: ', labels)

        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

## FOR NOW, LET'S ASSUME ALL IMAGES ARE OF SIZE 512x128

class DemoChordCnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # input shape: 3 x 128 x 512
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)), # output: 64 x 64 x 128

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)), # output: 128 x 32 x 32

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4), # output: 256 x 8 x 8

            nn.Flatten(), 
            nn.Linear(256*8*8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 5)
        )
                
    
    def forward(self, xb):
        out = self.network(xb)
        #print('out shape: ', out.shape)
        return out

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

device = get_default_device()
model = to_device(DemoChordCnnModel(), device)

train_device_dl = DeviceDataLoader(train_dl, device)
val_device_dl = DeviceDataLoader(val_dl, device)

for img, label in train_device_dl:
  print('img shape: ', img.shape)
  print('label: ', label)
  # img = torch.zeros(1,3,64,64)
  # to_device(img, device)
  # print(img.shape)
  # img = img[:,:,:64,:64]
  # img.to(device)
  # print('new img shape: ', img.shape)
  pred = model(img)
  print('pred:\n', pred.shape)
  break

num_epochs = 20
opt_func = torch.optim.SGD
lr = 0.001
history = fit(num_epochs, lr, model, train_device_dl, val_device_dl, opt_func)

num_epochs = 100
opt_func = torch.optim.Adam
lr = 0.001
history = fit(num_epochs, lr, model, train_device_dl, val_device_dl, opt_func)

print(index_to_label)
for img, label in train_device_dl:
    im = img[0,:,:,:]    
    im = im.permute(1,2,0).to('cpu')
    plt.imshow(im);plt.show()
    print('label: ', index_to_label[label.item()])
    pred = torch.argmax(model(img))
    print('prediction: ', index_to_label[pred.item()])

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
print('input: ', input)
print('target: ', target)
output = loss(input, target)

# #Cell to change images' sizes to fit the model

# import cv2
# train_dir = "C:/Users/David/Documents/huji/year 5/guitar_project/venv/demo_dataset/train"
# os.chdir(train_dir)
# for folder in os.listdir(train_dir):
#     os.chdir(train_dir + "/" + folder)
#     for file in os.listdir():
#         #location = data_dir+"/train/"+folder+"/"+file
#         img = cv2.imread(file)
#         resized_img = cv2.resize(img, (32,32))
#         print('resized_img shape: ' , resized_img.shape)
#         cv2.imwrite(file, resized_img)
#         plt.imshow(img)
#         #print(file)
#     os.chdir(train_dir)
for p in model.parameters():
  print(p)
  break

for p in model.parameters():
  print(p)
  break

model.parameters

model[1]

git status

