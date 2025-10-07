import numpy  as np  
import pandas as pd
import cv2
from tqdm import tqdm
import torch
import random
from sentence_transformers import SentenceTransformer
import inspect
import statistics

def set_random_seed(seed=42):
  """
  Fixed seeds for reproducibility
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
    
def read_images(image_paths):
  """
  Reads multiple images into a NumPy array stack using OpenCV.
  """
  images = []
  for img_path in tqdm(image_paths):
    img = cv2.imread(img_path)
    if img is not None:
      images.append(img)
  return np.stack(images)
    
def process_data(df):
    df['NumTags'] = df['tags'].apply(eval).apply(len)
    df['categories'] = df['categories'].apply(eval)
    df['Likes_Dislikes'] = df['Likes'] - df['Dislikes']
    return df
    
def get_normalize_images(images):
    """
    Calculate the mean and standard deviation of a dataset of images for normalization.

    The images are first scaled to the range [0, 1] by dividing by 255.
    """
    # Calculate mean and std over all images and pixels
    means = np.mean(images, axis=(0, 1, 2))  # Shape: (3,)
    stds = np.std(images, axis=(0, 1, 2))    # Shape: (3,)
    return (means,stds)
    
def get_object_info(obj):
    """
    Get comprehensive information about any object including attribute values
    """
    info = {
        'class': obj.__class__.__name__ if hasattr(obj, '__class__') else 'N/A',
        'module': getattr(obj, '__module__', 'N/A'),
        'attributes': {}
    }
    
    init_signature = inspect.signature(obj.__class__.__init__)
    init_params = list(init_signature.parameters.keys())[1:]  # Remove 'self'
    
    for param in init_params:
        if hasattr(obj, param):
            try:
                info['attributes'][param] = getattr(obj, param)
            except:
                info['attributes'][param] = "<Unable to read>"
    
    return info

def get_column_transformer_info(column_transformer):
    """
    Get comprehensive info for a ColumnTransformer
    """
    
    def get_transformer_info(transformer):
        return {
            'class': transformer.__class__.__name__,
            'module': transformer.__module__,
        }
    
    ct_info = {
        'type': 'ColumnTransformer',
        'transformers': [],
    }
    
    for name, transformer, columns in column_transformer.transformers:
        ct_info['transformers'].append({
            'name': name,
            'columns': columns,
            'transformer': get_transformer_info(transformer)
        })
    
    return ct_info
    
def best_trial_scores_ML(study):
    model = study.best_trial.user_attrs['model']['name']
    train_score = round(statistics.mean(study.best_trial.user_attrs['train_score']), 3)
    val_score = round(statistics.mean(study.best_trial.user_attrs['val_score']), 3)
    return model, train_score, val_score