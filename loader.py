#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
import os
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.utils.data as D
from torchvision import transforms


# # Dataset Class

# In[2]:


class ImageDS(D.Dataset):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(-2.1120, -2.0382, -1.8050), std=(0.0031, 0.0035, 0.0036))
    ])

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = f'/storage/aptosplus/{self.df.loc[idx, "ds"]}/{self.df.loc[idx, "id_code"]}.png'
        image = Image.open(path)
        label = torch.tensor(self.df.loc[idx, 'diagnosis'])

        return ImageDS.normalize(image), label


# In[6]:


class ImageDS_Stats(D.Dataset):
    normalize = transforms.Compose([
        transforms.ToTensor(),
#         transforms.Normalize(mean=(123.68, 116.779, 103.939), std=(58.393, 57.12, 57.375))
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

        return ImageDS.normalize(image), label


# In[ ]:




