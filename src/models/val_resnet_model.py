import torch

import numpy as np

from torch.autograd import Variable


from .utils import *

def validation(model, test_, device='cpu'):
    model.eval()
    test_iter=0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_):
            data, target = cus_aug(Variable(data.to(device))), Variable(target.to(device)) 
            output = model(data)
            _, pred = torch.max(output, 1)
            pred = output.data.cpu().numpy()
            gt = target.data.cpu().numpy()
            if test_iter==0:
                all_pred=pred
                all_gt=gt
            else:
                all_pred=np.vstack((all_pred,pred))
                all_gt  =np.vstack((all_gt,gt))

            test_iter=test_iter+1
        acc=accuracy(all_gt.reshape(all_gt.shape[0] * all_gt.shape[1]),all_pred)
        model.train()
        return acc