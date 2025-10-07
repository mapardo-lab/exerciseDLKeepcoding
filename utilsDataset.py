import torch
import numpy  as np  
import pandas as pd
from torch.utils.data import Dataset
import cv2
from PIL import Image
import os

class features_Dataset(Dataset): 
    """
    Class for handling feature related to engagement
    """
    def __init__(self, df, transform_features, transform_target):
        self.data = df
        self.transform_features = transform_features
        self.transform_target = transform_target
        self.features = transform_features.transform(df)
        self.target  = self.transform_target.transform(self.data)
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        # features 
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        # target
        target = torch.tensor(self.target[idx], dtype=torch.long)
        # result
        result = {'features': features, 'target': target}
        return result

class images_Dataset(Dataset): 
    """
    Class for handling images related to engagement
    """
    def __init__(self, df, transform_images, transform_target):
        self.data = df
        self.transform_images = transform_images
        self.target  = transform_target.transform(self.data)
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        # images
        images_transformed = self.transform_images.transform(self.data.iloc[idx])
        # target
        target = torch.tensor(self.target[idx], dtype=torch.long)
        # result
        result = {'images': images_transformed, 'target': target}
        return result

class multimodal3_Dataset(Dataset): 
    """
    Class for handling images related to engagement
    """
    def __init__(self, df, transform_images, transform_embeddings, transform_metadata, transform_target):
        self.data = df
        self.transform_images = transform_images
        self.metadata  = transform_metadata.transform(self.data)
        self.embeddings  = transform_embeddings.transform(self.data)
        self.target  = transform_target.transform(self.data)
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        # images
        images_transformed = self.transform_images.transform(self.data.iloc[idx])
        # embeddings 
        embeddings = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        # metadata
        metadata = torch.tensor(self.metadata[idx], dtype=torch.float32)
        # target
        target = torch.tensor(self.target[idx], dtype=torch.long)
        # result
        result = {'images': images_transformed, 'embeddings': embeddings, 'metadata': metadata, 'target': target}
        return result
        