#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import optuna
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import recall_score, precision_score, f1_score

from utilsFT import TargetFeature, MultiLabelBinarizerWrapper
from utils import set_random_seed, build_scorer, get_column_transformer_info
from utilsProc import process_data 
from utilsDataset import features_Dataset
from utilsNN import FCNN_shallow
from utilsOptuna import ObjectiveFunctionDL, create_study

def main():
    ## Check if run_name argument is provided
    if len(sys.argv) < 2:
        print("Error: No run name provided")
        print("Usage: script.py <run_name>")
        sys.exit(1)
    run_name = sys.argv[1]
    
    ## Random reproducibility
    set_random_seed()
    
    ## Load dataset
    data_file = "poi_dataset.csv"
    poi_data = pd.read_csv(data_file) 

    ## Proprocess data to train/validate model
    # simple process features + create new features
    poi_data_processed = process_data(poi_data) 
    
    # split data into train and test datasets
    test_size_test = 0.2
    df_train, df_test = train_test_split(poi_data_processed, test_size = test_size_test) 
    print(f'Number of samples')
    print(f'Train dataset: {df_train.shape[0]}')
    print(f'Test dataset: {df_test.shape[0]}')
    
    # preprocess data Imputation/Encoding/Transformation
    proc_features = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), ['xps', 'locationLon', 'locationLat', 'NumTags']),
            ('categories', MultiLabelBinarizerWrapper(), ['categories']), 
            ('tier',OneHotEncoder(sparse_output=False),['tier'])
        ],
        remainder="drop"
    )
    proc_features.fit(df_train)
    
    # split train data into train and validation datasets
    test_size_val = 0.2
    df_train, df_val = train_test_split(df_train, test_size = test_size_val) # *

    # Dataset
    train_dataset = features_Dataset(df_train, transform_features = proc_features)
    val_dataset = features_Dataset(df_val, transform_features = proc_features)
    
    # hyperparameter to optimizate
    search_space = {
        'data_loader': {
            'batch_size': {'type': 'int', 'low': 2, 'high': 5, 'exp2': True},
        },
        'model': {
            'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 1.0}
        },
        'optimizer': {
            'lr': {'type': 'float', 'low': 5e-4, 'high': 5e-1, 'log': True},
        },
        'criterion': {}
    }

    # fixed hyperparameters
    fixed_params = {
        'data_loader': {},
        'model': {
            'input_size': 20, 
            'num_classes': 2, 
        },
        'optimizer': {},
        'criterion': {}
    }

    # scores output
    scoring = {
        'sensitivity': build_scorer(recall_score, pos_label=1),
        'precision': build_scorer(precision_score, pos_label=1, zero_division = 0),
        'f1_score': build_scorer(f1_score, pos_label=1),
        'macro_precision': build_scorer(precision_score, average='binary', zero_division = 0)
    }

    # objetive function DL
    objective = ObjectiveFunctionDL(
        train_dataset = train_dataset, val_dataset = val_dataset,
        model = FCNN_shallow,
        criterion = CrossEntropyLoss,
        optimizer = Adam,
        num_epochs = 10,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
        fixed_params=fixed_params,
        search_space=search_space,
        scoring=scoring,
        score = 'f1_score',
        run_name=run_name
    )
    # parameters for study
    study_params = {
        'direction': 'maximize',
        'storage': 'sqlite:///optuna_DL_exercise.db',  # Persistent storage
        'study_name': sys.argv[0].replace('.py','').replace('./',''),
        'sampler': optuna.samplers.TPESampler(
            n_startup_trials = 10,
            n_ei_candidates = 24,
            seed=42),
        'pruner': optuna.pruners.MedianPruner(
            n_startup_trials = 5,
            n_warmup_steps = 3
        ),
        'load_if_exists': True  # Continue if study exists
    }

    # user attributes for study
    user_attr = [
        ('script', f'{sys.argv[0]}'),
        ('dataset', f'{data_file}'),
        ('preproc_data', f'{type(process_data)}'),
        ('split_test', {'test_size': test_size_test}),
        ('proc_features', get_column_transformer_info(proc_features)),
        ('split_val', {'test_size': test_size_val}),
        ('comments', 'Shallow FCNN using metadata')
    ]

    # create study
    study = create_study(study_params, user_attr)

    ## run trials
    study.optimize(objective, n_trials=10)
    print(f"Completed {len(study.trials)} trials")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

if __name__ == "__main__":
    main()