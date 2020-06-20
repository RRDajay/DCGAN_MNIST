# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:51:42 2020

@author: ryan_
"""

import torch 
import torch.nn as nn

class Discriminator(nn.Module):
  
  def __init__(self, channels_img, features_d):
    super().__init__()

    self.net = nn.Sequential(
        
        nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
       
        nn.Conv2d(features_d, features_d*2, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(features_d*2),
        nn.LeakyReLU(0.2),
        
        nn.Conv2d(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(features_d*4),
        nn.LeakyReLU(0.2),
        
        nn.Conv2d(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(features_d*8),
        nn.LeakyReLU(0.2),
        nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
        
        nn.Sigmoid()
    )

  def forward(self, x):
      return self.net(x)

class Generator(nn.Module):

   def __init__(self, channels_noise, channels_img, features_g):
    super().__init__()
    self.net = nn.Sequential( # 256, 1  16
        nn.ConvTranspose2d(channels_noise, features_g*16, kernel_size=4, stride=1, padding=0),
        
        nn.BatchNorm2d(features_g*16),
        nn.ReLU(),

        nn.ConvTranspose2d(features_g*16, features_g*8, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(features_g*8),
        nn.ReLU(),

        nn.ConvTranspose2d(features_g*8, features_g*4, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(features_g*4),
        nn.ReLU(),

        nn.ConvTranspose2d(features_g*4, features_g*2, kernel_size=4, stride=2, padding=1, ),
        nn.BatchNorm2d(features_g*2),
        nn.ReLU(),

        nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
        nn.Tanh()
    )

  
   def forward(self, x):
    
    return self.net(x)