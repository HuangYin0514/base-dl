import torch.nn as nn
from torchvision import models


class Resnet18Custom(nn.Module):
    def __init__(self):
        super(Resnet18Custom, self).__init__()

        self.layer1 = models.resnet18(pretrained=True)
        num_ftrs = self.layer1.fc.in_features
        self.layer1.fc = nn.Linear(num_ftrs, 2)


    def forward(self, x):
        out = self.layer1(x)
        return out
