import numpy  as np  
import pandas as pd
import cv2
from tqdm import tqdm

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