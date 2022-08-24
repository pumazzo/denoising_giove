# -*- coding: utf-8 -*-
# Data transformation and augmentation: work in progress¶

import numpy as np
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.utils import make_grid


import torchvision.transforms as T



class ToTensor_torch(nn.Module):

    def __init__(self):
        super().__init__()
        
    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            image = torch.from_numpy(x)
            image_cat = torch.cat((image.real,image.imag),dim=1)
            return image_cat

class norm_tensor(nn.Module):

    def __init__(self):
        super().__init__()
        
    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            mean, std = x.mean([1,2]), x.std([1,2])
            im = T.functional.normalize(x,mean, std)
            return im


class ToTensor(nn.Module):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # from numpy to tensor with channel stacked
        
        
        image = torch.from_numpy(image)
        image_cat = torch.cat((image.real,image.imag),dim=0)
        
        
        
        return image_cat

class AddNoise_torch(nn.Module):
    """add gaussian complex noise to sample and label to reduce overfitting"""
    def __init__(self, noise_std):
        super().__init__()
        self.noise_std=noise_std
        
    def forward(self, x):
        # noise added 
        
        with torch.no_grad():
            image=x[0]
            label=x[1]
            image = image + self.noise_std*image.std()*torch.randn(image.shape,dtype=image.dtype,device=image.device)
            label = label + self.noise_std*label.std()*torch.randn(label.shape,dtype=label.dtype,device=label.device)

        return image,label


class FreqToSpace(nn.Module):

    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = torch.complex(x[:,0,:,:], x[:,1,:,:])#f2s
            im = torch.fft.fftshift( torch.fft.ifft2( torch.fft.ifftshift(z,dim=[-2,-1] ),norm="ortho" ),dim=[-2,-1])
            im = torch.unsqueeze(im,dim=1)
            im = torch.cat((im.real,im.imag),dim=1)
            
            return im


class SpaceToFreq(nn.Module):

    def __init__(self):
        super().__init__()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = torch.complex(x[:,0,:,:], x[:,1,:,:])
            z = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(x,dim=[-1,-2]),norm="ortho"),dim=[-1,-2])
            z = torch.unsqueeze(z,dim=1)
            z = torch.cat((z.real,z.imag),dim=1)
            return z

class concatenateImageAndLabels(nn.Module):
    """To apply the same transformation in both image and labels the two arrays are stacked, transformed and then splitted back"""

    def __init__(self):
        super().__init__()
    

    def forward(self, x): 
        with torch.no_grad():
            z=torch.cat((x[0],x[1]),dim=0)
            return z

class splitImageAndLabels(nn.Module):

    def __init__(self,batch_size):
        super().__init__()
        self.batch_size = batch_size
    

    def forward(self, z ): 
        with torch.no_grad():
            x,y=torch.chunk(z,2,dim=0)
            return x,y

class prob_Rcrop(nn.Module):

    def __init__(self,p):
        super().__init__()
        self.p = p
        self.rc =  T.RandomResizedCrop((80,256),scale=(0.9, 1.1),ratio=(0.85, 1.2),interpolation = T.functional.InterpolationMode.NEAREST)

    

    def forward(self, x ): 
        with torch.no_grad():
            if torch.rand(1) < self.p:
                x = self.rc(x)
                return x
            else:
                return x


class phase_multiplication(nn.Module):

    def __init__(self,p,angle):
        super().__init__()
        self.p = p
        self.angle = torch.tensor([angle])

    

    def forward(self, x ): 
        with torch.no_grad():
            if torch.rand(1) < self.p:
                self.angle = torch.rand(1)*self.angle
                phase_mul = torch.tensor([[torch.cos(self.angle), torch.sin(self.angle)], [-1.*torch.sin(self.angle), torch.cos(self.angle)]],device=x.device)
                einsum_prod = torch.einsum ('bcxy, cc -> bcxy', x, phase_mul)
                return x
            else:
                return x

class prob_PADcrop(nn.Module):

    def __init__(self,p):
        super().__init__()
        self.p = p
        self.rc =  T.RandomCrop([80,256], padding=None, pad_if_needed=True, fill=0, padding_mode='reflect')

    

    def forward(self, x ): 
        with torch.no_grad():
            if torch.rand(1) < self.p:
                x = self.rc(x)
                return x
            else:
                return x       

class prob_Affine(nn.Module):

    def __init__(self,p):
        super().__init__()
        self.p = p
        self.rc =  T.RandomAffine(degrees=(-10, 10), translate=(0.25, 0.25))

    

    def forward(self, x ): 
        with torch.no_grad():
            if torch.rand(1) < self.p:
                x = self.rc(x)
                return x
            else:
                return x                   
