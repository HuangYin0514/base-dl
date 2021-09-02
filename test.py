
import numpy as np
from models import *

if __name__ == '__main__':
    model = resnet50(pretrained=True)    
    print(model)