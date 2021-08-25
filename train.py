import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

from options.train_options import TrainOptions
from data import *
from models import *


# opt ==============================================================================
opt = TrainOptions().parse()

# env setting ==============================================================================
# Fix random seed
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)

# speed up compution
torch.backends.cudnn.benchmark = True

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# data ============================================================================================================
# data Augumentation
train_transforms = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


test_transforms = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# data loader
train_list = glob.glob(os.path.join(opt.train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(opt.test_dir, '*.jpg'))

train_data = CatAndDog(train_list, transform=train_transforms)
test_data = CatAndDog(test_list, transform=test_transforms)


train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=opt.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=opt.batch_size, shuffle=True)

# model ============================================================================================================
net = Resnet18Custom()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)

# criterion ============================================================================================================
criterion = nn.CrossEntropyLoss()

# optimizer ============================================================================================================
optimizer = optim.Adam(params=net.parameters(), lr=opt.lr)


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


        # print('Training Loss: {:.4f}'.format(epoch_loss))

    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(
        epoch+1, epoch_accuracy, epoch_loss))


if __name__ == '__main__':
    for epoch in range(opt.start_epoch, opt.start_epoch+opt.epoch_num):
        train(epoch)
