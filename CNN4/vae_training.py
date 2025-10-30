import os
import gc
import sys
import cv2
import glob
import math
import time
import tqdm
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from accelerate import Accelerator

from functools import partial
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
# from diffusers import AutoencoderKL

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

from transformers import get_cosine_schedule_with_warmup  # keep this
from torch.optim import AdamW

from colorama import Fore, Back, Style
r_ = Fore.RED
b_ = Fore.BLUE
c_ = Fore.CYAN
g_ = Fore.GREEN
y_ = Fore.YELLOW
m_ = Fore.MAGENTA
sr_ = Style.RESET_ALL

config = {'lr':1e-3,
        'wd':1e-2,
        # 'bs':256,
        'bs':16,
        'img_size':128,
        'epochs':100,
        'seed':1000}

def get_train_transforms():
    return A.Compose(
        [
            A.Resize(config['img_size'],config['img_size'],always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ])


class ImageNetDataset(Dataset):
    def __init__(self, dataset, augmentations):
        self.dataset = dataset
        self.augmentations = augmentations

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        return image, label

    def __len__(self):
        return len(self.dataset)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class PretrainedSDVAE(nn.Module):
    """
    AutoencoderKL wrapper with optional VAE KL term.
    Inputs x_01 in [0,1]; returns recon_01 in [0,1].
    MSE + beta*KL is computed in the native [-1,1] space.
    """
    def __init__(
        self,
        repo_id="stabilityai/sd-vae-ft-mse",
        train_vae=False,
        use_kl=True,          # <— turn KL on/off
        beta=1.0,             # <— weight for KL
        dtype=torch.float32
    ):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(repo_id, torch_dtype=dtype)
        self.vae.requires_grad_(train_vae)
        self.train_vae = train_vae
        self.use_kl = use_kl
        self.beta = beta
        self.scaling_factor = float(self.vae.config.scaling_factor)

    @staticmethod
    def to_range_minus1_1(x):  # [0,1] -> [-1,1]
        return x * 2.0 - 1.0

    @staticmethod
    def to_range_0_1(x):       # [-1,1] -> [0,1]
        return (x + 1.0) / 2.0

    def encode(self, x_01):
        x_m11 = self.to_range_minus1_1(x_01)
        posterior = self.vae.encode(x_m11).latent_dist
        latents = posterior.rsample() * self.scaling_factor  # keep reparam gradients
        return latents, posterior

    def decode(self, latents):
        latents = latents / self.scaling_factor
        x_hat_m11 = self.vae.decode(latents).sample
        x_hat_01 = self.to_range_0_1(x_hat_m11).clamp(0, 1)
        return x_hat_01

    @staticmethod
    def kl_from_posterior(posterior):
        # KL(q||p) with p=N(0,I), q=N(mu, diag(exp(logvar)))
        # 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
        return 0.5 * torch.sum(
            torch.exp(posterior.logvar) + posterior.mean**2 - 1.0 - posterior.logvar,
            dim=1
        ).mean()

    def forward(self, x_01, return_latents=True):
        # encode (works in [-1,1] internally)
        x_m11 = self.to_range_minus1_1(x_01)
        posterior = self.vae.encode(x_m11).latent_dist
        z = posterior.sample() * self.scaling_factor

        # decode
        z_dec = z / self.scaling_factor
        x_hat_m11 = self.vae.decode(z_dec).sample
        recon_01 = self.to_range_0_1(x_hat_m11).clamp(0, 1)

        # losses
        mse = nn.functional.mse_loss(x_hat_m11, x_m11, reduction="mean")  # compute in [-1,1]
        logs = {"mse": mse}

        if self.train_vae:
            if self.use_kl:
                kl = self.kl_from_posterior(posterior)  # your formula
                loss = mse + self.beta * kl
                logs["kl"] = kl
                logs["loss"] = loss
            else:
                loss = mse
                logs["loss"] = loss

        # diagnostics
        logs.update({
            "mu_mean": posterior.mean.mean().detach(),
            "logvar_mean": posterior.logvar.mean().detach(),
        })

        if self.train_vae:
            return recon_01, logs
        else:
            if return_latents:
                return recon_01, z
            return recon_01


class VAE(nn.Module):
    def __init__(self, latent_dim=512, img_size=128, beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.img_size = img_size

        # Encoder (pad=1 keeps exact halving)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(inplace=True),
        )
        feat_spatial = img_size // 4        # 128 -> 32
        enc_feat = 64 * feat_spatial * feat_spatial
        self.fc_mu    = nn.Linear(enc_feat, latent_dim)
        self.fc_logv  = nn.Linear(enc_feat, latent_dim)

        # Decoder: start from (32, S, S) where S = img_size//4
        self.dec_in = nn.Linear(latent_dim, 32 * feat_spatial * feat_spatial)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2),   # 32 -> 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # 64 -> 128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )

        # learned Gaussian scale (log std)
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def encode(self, x):
        h = self.enc(x)
        h = torch.flatten(h, 1)
        mu = self.fc_mu(h)
        logvar = self.fc_logv(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        feat_spatial = self.img_size // 4
        h = self.dec_in(z)
        h = h.view(z.size(0), 32, feat_spatial, feat_spatial)
        x_hat = self.dec(h)
        return x_hat

    @staticmethod
    def kl_from_mu_logvar(mu, logvar):
        # KL(q||p) with p=N(0,I), q=N(mu, diag(exp(logvar)))
        # 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)

    def gaussian_nll(self, x, x_hat):
        # log std is shared scalar; reconstruction NLL = -log p(x|z)
        scale = torch.exp(self.log_scale)
        dist = torch.distributions.Normal(loc=x_hat, scale=scale)
        # negative log-likelihood
        nll = -dist.log_prob(x).sum(dim=(1, 2, 3))
        return nll

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)

        rec = self.gaussian_nll(x, x_hat)      # shape [B]
        kl  = self.kl_from_mu_logvar(mu, logvar)  # shape [B]
        loss = (rec + self.beta * kl).mean()

        return x_hat, {"loss": loss, "rec": rec.mean(), "kl": kl.mean(), "log_scale": self.log_scale.detach()}


def evaluate(model,valid_loader):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i, (inputs, l_) in enumerate(valid_loader):
            inputs = inputs.to(model.device)
            x_hat, r = model(inputs)
            valid_loss += r["loss"].item()

    valid_loss /= len(valid_loader)
    return valid_loss
    
def train_and_evaluate_loop(train_loader,valid_loader,model,optimizer,
                            epoch,best_loss,lr_scheduler=None, device=None):
    train_loss = 0
    model = model.train()

    for inputs, _ in tqdm.tqdm(train_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        x_hat, r = model(inputs)
        r["loss"].backward()
        optimizer.step()

        train_loss += r["loss"].item()

        if lr_scheduler:
            lr_scheduler.step()
    
    train_loss /= len(train_loader)
    valid_loss = evaluate(model,valid_loader)

    if valid_loss <= best_loss:
        print(f"Epoch:{epoch} |Train Loss:{train_loss}|Valid Loss:{valid_loss}")
        print(f"{g_}Loss Decreased from {best_loss} to {valid_loss}{sr_}")

        best_loss = valid_loss
        torch.save(model.state_dict(),'./imagenet_vae_model.bin')
                
    return best_loss


def main():

    seed_everything(seed=config['seed'])

    accelerator = Accelerator()
    print(f"{accelerator.device} is used")

    device = accelerator.device

    model = VAE(latent_dim=512, img_size=128, beta=1.0).to(device)
    # model = PretrainedSDVAE(train_vae=True, use_kl=True, beta=1.0).to(device)

    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'],config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],
                             std=[0.5,0.5,0.5]),
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join("C:/Users/pzaka/Documents/datasets/imagewoof2", "train"), transform=train_transform)
    train_dl = DataLoader(train_dataset, batch_size=config['bs'], shuffle=True, num_workers=0)
    
    #valid
    valid_dataset = datasets.ImageFolder(root=os.path.join("C:/Users/pzaka/Documents/datasets/imagewoof2", "val"), transform=train_transform)
    valid_dl = DataLoader(valid_dataset, batch_size=config['bs'], shuffle=False, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=config['epochs'] * len(train_dl))

    model, train_dl, valid_dl, optimizer, lr_scheduler = accelerator.prepare(model, train_dl, valid_dl, optimizer, lr_scheduler)

    best_loss = 9999999
    start_time = time.time()
    for epoch in range(config["epochs"]):
        print(f"Epoch Started:{epoch}")
        best_loss = train_and_evaluate_loop(train_dl,valid_dl,model,optimizer,epoch,best_loss,lr_scheduler, device=device)
        
        end_time = time.time()
        print(f"{m_}Time taken by epoch {epoch} is {end_time-start_time:.2f}s{sr_}")
        start_time = end_time
        
    return best_loss


if __name__ == "__main__":
    main()