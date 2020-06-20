# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 22:39:47 2020

@author: ryan_
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from DCGAN import Discriminator, Generator


# Hyperparameter
learning_rate = 2e-4
batch_size = 64
image_size = 64
channels_img = 1
channels_noise = 256
num_epochs = 10

features_d = 16
features_g = 16

# Data Preprocessing and Data Loader 
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))
])

dataset = torchvision.datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

# GPU or CPU 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Initialization 
netD = Discriminator(channels_img, features_d).train(mode=True).to(device)
netG = Generator(channels_noise, channels_img, features_g).train(mode=True).to(device)

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss
criterion = nn.BCELoss() 

# Tensorboard init
writer_real = SummaryWriter(f'runs/GAN-MNIST/test-real')
writer_fake = SummaryWriter(f'runs/GAN-MNIST/test-fake')

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        
        data = data.to(device)
    
        ### Train Discriminator: max log(D(x)) + log(1-D(G(z)))    
        # Train discriminator with real data
        netD.zero_grad()
        output = netD(data).reshape(-1)
        real_loss = criterion(output, (torch.ones(data.shape[0])*0.5).to(device))
        D_x = output.mean().item()
        
        # Train discriminator with fake data
        noise = torch.randn(data.shape[0], channels_noise, 1, 1).to(device)    
        fake = netG(noise)
        output = netD(fake.detach()).reshape(-1)
        fake_loss = criterion(output, (torch.zeros(data.shape[0])*0.1).to(device))
        
        # Combine loss from real image and fake image data into 
        lossD = real_loss + fake_loss
        lossD.backward()
        optimizerD.step()
        
        ### Train Generator: max log(D(G(z)))
        netG.zero_grad()
        label = torch.ones(data.shape[0]).to(device)
        output = netD(fake).reshape(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch:{epoch+1}/{num_epochs}\t\tBatch:{batch_idx}/{len(dataloader)}\t\tloss D: {lossD:.4f}\t\tLoss G:{lossG:.4f}\t\tD(x):{D_x:.4f}")
            
            with torch.no_grad():
                fake = netG(torch.randn(64, channels_noise, 1, 1).to(device))
                img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image('Mnist-real-images', img_grid_real)
                writer_fake.add_image('Mnist-fake-imaages', img_grid_fake)
        
writer_real.close()
writer_fake.close()
 











