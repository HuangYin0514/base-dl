import os
import glob
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import datasets

from data import *
from models import *


# opt ==============================================================================
parser = argparse.ArgumentParser(description="Person ReID Frame")
# base
parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
# env setting
parser.add_argument("--random_seed", type=int, default="1")
# data
parser.add_argument("--img_height", type=int, default=12)
parser.add_argument("--img_width", type=int, default=12)
parser.add_argument(
    "--train_dir", type=str, default="./datasets/hymenoptera_data/train"
)
parser.add_argument("--test_dir", type=str, default="./datasets/hymenoptera_data/val")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)
# Optimizer
parser.add_argument("--lr", type=float, default=0.1)
# train
parser.add_argument("--start_epoch", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=1)
# parse
opt = parser.parse_args()

# env setting ==============================================================================
# Fix random seed
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)
# speed up compution
torch.backends.cudnn.benchmark = True
# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# data ============================================================================================================
# data Augumentation
train_transforms = T.Compose(
    [
        T.Resize(((opt.img_height, opt.img_width)), interpolation=3),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_transforms = T.Compose(
    [
        T.Resize(((opt.img_height, opt.img_width)), interpolation=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
# data loader'
train_dataset = datasets.ImageFolder(root=opt.train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=opt.test_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers
)
class_names = train_dataset.classes

# model ============================================================================================================
model = Resnet18Custom()
model = model.to(device)
if device == "cuda":
    model = torch.nn.DataParallel(model)

# criterion ============================================================================================================
criterion = F.cross_entropy

# optimizer ============================================================================================================
optimizer = optim.Adam(params=model.parameters(), lr=opt.lr)

# scheduler ============================================================================================================
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# save dir path ============================================================================================================

# Training and test ============================================================================================================


def train(epoch):
    # model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print("is ok !")


if __name__ == "__main__":
    train(1)
