import os
import glob
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets

from options.train_options import TrainOptions
from data import *
from models import *


# opt ==============================================================================
opt = TrainOptions().parse()
TrainOptions().print_options(opt)

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
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# data loader'
data_dir = './datasets/hymenoptera_data/'

train_dataset = datasets.ImageFolder(
    root=data_dir+'train', transform=train_transforms)
test_dataset = datasets.ImageFolder(
    root=data_dir+'val', transform=train_transforms)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                         shuffle=True, num_workers=4)

class_names = train_dataset.classes

# model ============================================================================================================
model = Resnet18Custom()
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)

# criterion ============================================================================================================
criterion = nn.CrossEntropyLoss()

# optimizer ============================================================================================================
optimizer = optim.Adam(params=model.parameters(), lr=opt.lr)


# scheduler ============================================================================================================
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# save dir path ============================================================================================================


# Training and test ============================================================================================================
def train():
    since = time.time()
    for epoch in range(1):
        # print('Epoch {}/{}'.format(epoch, opt.num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        model.train()

        running_loss = 0.0
        running_corrects = 0

        print('123')
        

    return model


train()
