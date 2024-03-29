import torch

import numpy as np

import matplotlib.pyplot as plt
import random

from IPython.display import clear_output
import time
import numpy as np


#import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable

from sklearn.metrics import accuracy_score
from skimage.transform import resize

from .val_resnet_model import validation
from .utils import *

from torch.optim.lr_scheduler import ReduceLROnPlateau



def train(net, train_, val_, criterion, optimizer, epochs=None, scheduler=None, weights=None, save_epoch = 10, device = 'cpu', save_path = './'):
    losses=[]; acc=[]; mean_losses=[]; val_acc=[]
    iter_ = t0 =0
    for e in range(1, epochs + 1):
        net.train()
        for batch_idx, (data, target) in enumerate(train_):
            data, target =  cus_aug(Variable(data.to(device))), Variable(target.to(device)),
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses = np.append(losses,loss.item())
            mean_losses = np.append(mean_losses, np.mean(losses[max(0,iter_-100):iter_]))
            if iter_ % 40 == 0: #vary how often ysou want to update plots
                clear_output()
                print('Iteration Number',iter_,'{} seconds'.format(time.time() - t0))
                t0 = time.time()
                pred = output.data.cpu().numpy()
                pred=sigmoid(pred)
                gt = target.data.cpu().numpy()
                acc = np.append(acc,accuracy(gt,pred))
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}\tLearning Rate:{}'.format(
                    e, epochs, batch_idx, len(train_),
                    100. * batch_idx / len(train_), loss.item(), acc[-1],optimizer.param_groups[0]['lr']))
                plt.plot(mean_losses) and plt.show()
                val_acc = np.append(val_acc,validation(net, val_, device))
                print('validation accuracy : {}'.format(val_acc[-1]))
                plt.plot( range(len(acc)) ,acc,'b',label = 'training')
                plt.plot( range(len(val_acc)), val_acc,'r--',label = 'validation')
                plt.legend() and plt.show()
            iter_ += 1
            del(data, target, loss)
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(mean_losses[-1])
            else:
                scheduler.step()

        if e % save_epoch == 0:
            
            torch.save(net.state_dict(),'{}Eurosat{}'.format(save_path,e))
    return net


