import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from PIL import Image
import math
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
from torchvision.models import alexnet, AlexNet_Weights
import torchvision.utils as vutils

from IPython.display import clear_output
import piq
from piq import ssim, multi_scale_ssim
import lpips

from pytorch_fid import fid_score
from torchmetrics.image import StructuralSimilarityIndexMeasure

torch.manual_seed(111)  # Random Generator Seed

def calculate_metrics(real, fake):
 
    metrics = {}

    # MSE
    metrics['MSE'] = F.mse_loss(fake, real).item()

    # MAE
    metrics['MAE'] = F.l1_loss(fake, real).item()

    # PSNR (range [-1,1] → MAX_I = 2)
    mse = metrics['MSE']
    metrics['PSNR'] = 20 * torch.log10(2.0 / torch.sqrt(torch.tensor(mse))).item()
   
    return metrics
#----------------------------------------------------------------
def evaluate_test_set(model, test_loader, device):
    model.eval()

    all_metrics = {
        "MSE": [],
        "MAE": [],
        "PSNR": []
    }

    with torch.no_grad():
        for real,_ in test_loader:

            real = real.to(device).float()

            # تولید تصویر fake
            fake = model(real)

            # اگر grayscale → 3 کاناله (برای consistency)
            if real.shape[1] == 1:
                real = real.repeat(1, 3, 1, 1)
                fake = fake.repeat(1, 3, 1, 1)

            metrics = calculate_metrics(real, fake)

            for key in all_metrics:
                all_metrics[key].append(metrics[key])

    # میانگین کل دیتاست
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")

    return avg_metrics