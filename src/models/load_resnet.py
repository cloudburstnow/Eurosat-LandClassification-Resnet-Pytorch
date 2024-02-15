
import torch
import numpy as np

import torch.nn as nn

import torchvision
from torchvision import models


def Load_model(resnet_val = 50):
    
    if resnet_val == 18:
        model_ft = models.resnet18()
    else if resnet_val == 50:
        model_ft = models.resnet50()

    
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)
    
    inBands = 3

    model_ft.conv1 = nn.Conv2d(inBands, 64, kernel_size=7, stride=2, padding = 3, bias = False)

    print('Model Loaded')

    return model_ft.to(device)