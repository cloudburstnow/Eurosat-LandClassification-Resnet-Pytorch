
import torch
import numpy as np

from skimage import io

import torchvision
from torchvision import models

from sklearn.model_selection import train_test_split



def img_loader(path):
    image = np.asarray((io.imread(path))/32000,dtype='float32')
    return image.transpose(2,0,1)

def make_dataloaders(loader_func = img_loader, batch_size_val = 128, data_path):
        
    data = torchvision.datasets.DatasetFolder(root=data_path,loader = loader_func, transform=None, extensions = 'jpg')
    train_set, val_set = train_test_split(data, test_size = 0.2, stratify = data.targets)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_val, shuffle=True, num_workers=3, drop_last = True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_val, shuffle=True, num_workers=0, drop_last = True)

    return train_loader, val_loader