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

def build_scorer(metric_func, **metric_params):
    """
    Create a scorer function with fixed parameters (f1_score, recall_score, ...)
    """
    def scorer(y_true, y_pred):
        return metric_func(y_true, y_pred, **metric_params)
    return scorer

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

def info_object(obj):
    results = {}
    results['name'] = obj.__name__
    results['module'] = obj.__module__
    return results