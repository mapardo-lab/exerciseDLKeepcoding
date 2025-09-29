
#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from utilsClass import TargetFeature, MultiLabelBinarizerWrapper
from utils import process_data, set_random_seed
from utilsDataset import meta_Dataset
from utilsNN import FCNN
from utilsTrain import train_model

def main():
    ### Random reproducibility

    ### Load dataset
    
    ### Prepare data to train/validate model
    # simple process features + create new features
    # split data into train and test datasets
    # preprocess features Imputation/Encoding (target + explanatory)

    ### Configure optimization
    # objetive function for Optuna
        # hyperparameters to optimize
        # name run
        # build model 
        # train/validate model
        # return score 
    # create/load study 
    # set user_attr

    ### Run trials
    