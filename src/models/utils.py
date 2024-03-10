
import torch
import random
from sklearn.metrics import accuracy_score
from skimage.transform import resize

import warnings

import numpy as np

def cus_aug(data):
    data = torch.rot90(data,random.randint(-3,3), dims=random.choice([[3,2],[2,3]]))
    if random.random()>0.75:
        data = torch.flip(data, dims = random.choice([[2,],[3,],[2,3]]))
    pixmis = torch.empty_like(data).random_(data.shape[-1])
    pixmis = torch.where(pixmis>(data.shape[-1]/8),torch.ones_like(data),torch.zeros_like(data))
    return data* pixmis

def get_random_pos(img, window_shape = [55,55] ):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    y1 = random.randint(0, H - h - 1)

    return x1, x1 + w, y1, y1 + h #x1, x2, y1, y2

def random_crop_area(img):
    x1,x2,y1,y2 = get_random_pos(img)
    Sen_Im = img[:, x1:x2,y1:y2]
    return resize(Sen_Im,img.shape,anti_aliasing=True)



def sigmoid(z):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning) # Overflows occur in the exp when a z value is between # -89 and 100, due to the floating point precision of float32
        z = np.clip(z, -100, 100)
        return 1 / (1 + np.exp(-z))


def accuracy(gt_S,pred_S):       
    _, alp = torch.max(torch.from_numpy(pred_S), 1)
    return accuracy_score(gt_S,np.asarray(alp)) 