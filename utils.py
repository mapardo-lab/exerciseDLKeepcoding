import numpy  as np  
import pandas as pd
import cv2
from tqdm import tqdm
import torch
import random

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