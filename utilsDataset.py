import torch
import numpy  as np  
from torch.utils.data import Dataset

class features_Dataset(Dataset): 
    """
    Class for handling features and target
    """
    def __init__(self, df, transform_features, transform_target = None):
        self.data = df
        self.transform_features = transform_features
        self.features = transform_features.transform(df)
        if transform_target is not None:
            self.target = transform_target.transform(self.data)
        elif 'target' in df.columns:
            self.target = np.array(self.data['target'])
        else:
            self.target = np.zeros(df.shape[0])
    
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
    Class for handling images and target
    """
    def __init__(self, df, transform_images, transform_target=None):
        self.data = df
        self.transform_images = transform_images
        if transform_target is not None:
            self.target = transform_target.transform(self.data)
        elif 'target' in df.columns:
            self.target = np.array(self.data['target'])
        else:
            self.target = np.zeros(df.shape[0])
    
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
    Class for handling images, metadata, embeddings and target
    """
    def __init__(self, df, transform_images, transform_embeddings, transform_metadata, transform_target=None):
        self.data = df
        self.transform_images = transform_images
        self.metadata  = transform_metadata.transform(self.data)
        self.embeddings  = transform_embeddings.transform(self.data)
        if transform_target is not None:
            self.target = transform_target.transform(self.data)
        elif 'target' in df.columns:
            self.target = np.array(self.data['target'])
        else:
            self.target = np.zeros(df.shape[0])
    
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
        