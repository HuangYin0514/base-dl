
import numpy as np
import torch
from models import *

if __name__ == '__main__':
    model = Resnet_Classification()    
    inputs = torch.randn(3,3,224,224)
    outputs = model(inputs)
    print(model)
    print(outputs)