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
  def __init__(self, target, features, transform=None):
    self.target = torch.tensor(target)
    self.features = torch.tensor(features, dtype=torch.float32)
    self.transform = transform

  def __len__(self):
    return len(self.target)

  def __getitem__(self, idx):
    features = self.features[idx]
    target = self.target[idx]
    result = {'meta': features, 'target': target}
    return result

class img_Dataset(Dataset):
  """
  Class for handling images and feature related to engagement
  """
  def __init__(self, target, image_path, transform_img=None, transform_target=None):
    self.target = torch.tensor(target)
    self.image_path = image_path
    self.transform_img = transform_img
    self.transform_target = transform_target

  def __len__(self):
    return len(self.target)

  def __getitem__(self, idx):
#    print(type(self.image_path))
#    print(self.image_path.shape)
#    print(self.image_path[idx])
    image = cv2.imread(self.image_path[idx])
    target = self.target[idx]
    if self.transform_img is not None:
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      pil_image = Image.fromarray(image_rgb)
      image_transformed = self.transform_img(pil_image)
    if self.transform_target is not None:
      target = self.transform_target(target)
    result = {'img': image_transformed, 'target': target}
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