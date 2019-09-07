#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
import os
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.utils.data as D
from torchvision import transforms


# # Dataset Class

# In[6]:


class ImageDS(D.Dataset):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(794.9223, 685.4205, 605.1915), std=(58.8981, 57.5124, 57.9897))
    ])

    def __init__(self, dx, dy, small=False):
        self.dx = dx
        self.dy = dy
        self.small = 'sm' if small else ''

    def __len__(self):
        return len(self.dx)

    def __getitem__(self, idx):
        path = f'/storage/aptosplus/{self.dx.iloc[idx]["ds"]}{self.small}/{self.dx.iloc[idx]["id_code"]}.png'
        image = Image.open(path)

        return ImageDS.normalize(image), torch.tensor(self.dy.iloc[idx], dtype=torch.float)


# In[7]:


class ImageDS_Stats(D.Dataset):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = f'/storage/aptosplus/{self.df.loc[idx, "ds"]}/{self.df.loc[idx, "id_code"]}.png'
        image = Image.open(path)
        image = image.resize((512, 512))
        label = torch.tensor(self.df.loc[idx, 'diagnosis'])

        return ImageDS.transform(image), label

