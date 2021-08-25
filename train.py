import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

from options.train_options import TrainOptions
from data.cat_dog import CatAndDog
from models import *



# opt ==============================================================================
opt = TrainOptions().parse()

# env setting ==============================================================================
# Fix random seed
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# speed up compution
torch.backends.cudnn.benchmark = True

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# data ============================================================================================================
# data Augumentation
train_transforms =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


test_transforms = transforms.Compose([   
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

# data loader
train_dir = './datasets/dogVScat/train'
test_dir = 'datasets/dogVScat/test'
batch_size=64

train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

train_data = CatAndDog(train_list, transform=train_transforms)
test_data = CatAndDog(test_list, transform=test_transforms)


train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

# model ============================================================================================================
net = Cnn()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)

# criterion ============================================================================================================
criterion = nn.CrossEntropyLoss()

# optimizer ============================================================================================================
optimizer = optim.Adam(params = net.parameters(),lr=0.001)


# scheduler ============================================================================================================
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# save dir path ============================================================================================================


# Training and test ============================================================================================================
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    epoch_loss = 0
    epoch_accuracy = 0
    
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        
        output = net(data)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)
        print(epoch_loss)
        
    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))


def test(epoch):
    net.eval()
    for data, fileid in test_loader:
        data = data.to(device)
        preds = net(data)
        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
    

if __name__ == '__main__':
    epoch = 1
    for epoch in range(epoch, epoch+200):
        train(epoch)
        test(epoch)
        # scheduler.step()
    