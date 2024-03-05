import os
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import transforms as T
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm, Flatten, Conv2d
import torchvision
from torchvision import transforms, models
import math
import copy
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
import GPUtil
import nibabel as nib
import numpy as np
from tqdm import tqdm, trange
from itertools import cycle
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm
from math import exp, sqrt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as cal_PSNR
from skimage.metrics import structural_similarity as cal_SSIM
import wandb
from torchvision.utils import save_image
import random
import torchvision.transforms.functional as FF


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default="0")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=250)
parser.add_argument('--lr_age', type=float, default=0.001)
parser.add_argument('--lr_gan', type=float, default=0.0005)
parser.add_argument('--lr_map', type=float, default=0.00001)
parser.add_argument('--lr_id', type=float, default=0.00001)
parser.add_argument('--scheduler', type=str, default="yes")
parser.add_argument('--id_optim', type=str, default="Adam")
args = parser.parse_args()





class L2Reconloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, output, weights):
        n_samples = input.shape[0]
        total_loss = 0
        for i in range(n_samples):
            loss = nn.MSELoss()(input[i], output[i])
            loss = weights[i] * loss
            total_loss += loss

        return total_loss / len(output)


def compute_cosine_weights(x):
	""" Computes weights to be used in the id loss function with minimum value of 0.5 and maximum value of 1. """
	values = np.abs(x.cpu().detach().numpy())
	# assert np.min(values) >= 0. and np.max(values) <= 1., "Input values should be between 0. and 1!"
	weights = 0.25 * (np.cos(np.pi * values)) + 0.75
	return weights





def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



def D_orth_2D(out, c4, c3, c2, c1, id): # cosine similarity to zero for orthogonality

    d_c1 = nn.MaxPool2d(kernel_size=16)(c1)
    d_c2 = nn.MaxPool2d(kernel_size=8)(c2)
    d_c3 = nn.MaxPool2d(kernel_size=4)(c3)
    d_c4 = nn.MaxPool2d(kernel_size=2)(c4)

    age = torch.cat((d_c1, d_c2, d_c3, d_c4, out), 1)
    age = torch.mean(age, dim=1, keepdim=True)

    return torch.abs(F.cosine_similarity(age.detach(), id, dim=-1).mean())



def age_to_onehot(age):
    # Subtract the minimum age (48 in this case) to make the ages start from 0
    age_adjusted = age - 48
    one_hot = F.one_hot(age_adjusted, num_classes=33)
    one_hot = one_hot.view(-1, 33)

    return one_hot




"""
Utilities for SFCN
"""

def my_KLDivLoss(x, y):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y += 1e-16
    n = y.shape[0]
    loss = loss_func(x, y) / n
    #print(loss)
    return loss


def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers