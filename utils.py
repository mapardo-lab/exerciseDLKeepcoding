import numpy  as np  
import torch
import random
import dill
import base64
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

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

def info_train(train_config):
    result = {}
    for key, value in train_config.items():
        if (isinstance(value, type) | callable(value)):
            result[key] = info_object(value)
    return result

def serial_encode(obj):
    serialized = dill.dumps(obj)
    encoded = base64.b64encode(serialized).decode("utf-8")
    return encoded

def serial_decode(encoded):
    decoded = dill.loads(base64.b64decode(encoded))
    return decoded

def confusion_matrix_list(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).tolist()

def save_objects(obj_dict, filename):
    """Save multiple objects to a file"""
    with open(filename, 'wb') as f:
        pickle.dump(obj_dict, f)

def load_objects(filename):
    """Load objects from a file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(filename + ' does not exist.')
    
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
