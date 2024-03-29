import torch.nn as nn
from torchvision import models


def Load_model(resnet_val=50, device = "cpu"):

    if resnet_val == 18:
        model_ft = models.resnet18()
    elif resnet_val == 50:
        model_ft = models.resnet50()

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)

    inBands = 3

    model_ft.conv1 = nn.Conv2d(
        inBands, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    print("Model Loaded")

    return model_ft.to(device)
