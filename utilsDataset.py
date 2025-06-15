import torch
import numpy  as np  
import pandas as pd
from torch.utils.data import Dataset
import cv2
from PIL import Image
import os

class meta_Dataset(Dataset):
  """
  Class for handling metadata features and feature related to engagement
  """
  def __init__(self, engagement, features, transform=None):
    self.engagement = torch.tensor(engagement)
    self.features = torch.tensor(features.values, dtype=torch.float32)
    self.transform = transform

  def __len__(self):
    return len(self.engagement)

  def __getitem__(self, idx):
    features = self.features[idx]
    engagement = self.engagement[idx]
    result = {'meta': features, 'target': engagement}
    return result

class img_Dataset(Dataset):
  """
  Class for handling images and feature related to engagement
  """
  def __init__(self, engagement, image_path, transform=None):
    self.engagement = torch.tensor(engagement.values)
    self.image_path = image_path
    self.transform = transform

  def __len__(self):
    return len(self.engagement)

  def __getitem__(self, idx):
    image = cv2.imread(os.path.join(self.image_path.iloc[idx]))
    if self.transform is not None:
      image = self.transform(image)
    engagement = self.engagement[idx]
    result = {'img': image, 'target': engagement}
    return result

class img_meta_Dataset(Dataset):
  """
  Class for handling metadata features, images and feature related to engagement
  """
  def __init__(self, engagement, image_path, features, transform=None):
    self.engagement = torch.tensor(engagement.values)
    self.features = torch.tensor(features.values, dtype=torch.float32)
    self.image_path = image_path
    self.transform = transform

  def __len__(self):
    return len(self.engagement)

  def __getitem__(self, idx):
    image = cv2.imread(os.path.join(self.image_path.iloc[idx]))
    if self.transform is not None:
      image = self.transform(image)
    engagement = self.engagement[idx]
    features = self.features[idx]

    result = {'img': image, 'meta': features, 'target': engagement}
    return result

class img_meta_resnet_Dataset(Dataset):
  """
  Class for handling metadata features, images for ResNet 
  and feature related to engagement
  """
  def __init__(self, engagement, image_path, features, transform=None):
    self.engagement = torch.tensor(engagement.values)
    self.features = torch.tensor(features.values, dtype=torch.float32)
    self.image_path = image_path
    self.transform = transform

  def __len__(self):
    return len(self.engagement)

  def __getitem__(self, idx):
    image = Image.open(os.path.join(self.image_path.iloc[idx]))
    if self.transform is not None:
      image = self.transform(image)
    engagement = self.engagement[idx]
    features = self.features[idx]

    result = {'img': image, 'meta': features, 'target': engagement}
    return result