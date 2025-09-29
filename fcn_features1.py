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
    ## Check if run_name argument is provided
    if len(sys.argv) < 2:
        print("Error: No run name provided")
        print("Usage: script.py <run_name>")
        sys.exit(1)
    run_name = sys.argv[1]
    
    ## Random reproducibility
    set_random_seed()
    
    ## Load dataset
    poi_data = pd.read_csv("poi_dataset.csv")

    ## Prepare data to train/validate model
    # simple process features + create new features
    poi_data_processed = process_data(poi_data)
    
    # split data into train and test datasets
    df_train, df_test = train_test_split(poi_data_processed, test_size = 0.2, random_state = 42)
    print(f'Number of samples.')
    print(f'Train dataset: {df_train.shape[0]}')
    print(f'Test dataset: {df_test.shape[0]}')
    
    # preprocess features Imputation/Encoding (target + explanatory)
    preproc_target = TargetFeature(col1_name='Visits', col2_name='Likes_Dislikes')
    preproc_explanatory = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), ['xps', 'locationLon', 'locationLat', 'NumTags']),
            ('categories', MultiLabelBinarizerWrapper(), ['categories']), 
            ('tier',OneHotEncoder(sparse_output=False),['tier'])
        ],
        remainder="drop"
    )
    
    X_train = preproc_explanatory.fit_transform(df_train)
    y_train = preproc_target.fit_transform(df_train)
    X_test = preproc_explanatory.transform(df_test)
    y_test = preproc_target.transform(df_test)
    
    # split train data into train and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

    ## Configure optimization
    def objective(trial):
        
        train_dataset = meta_Dataset(y_train, X_train)
        val_dataset = meta_Dataset(y_val, X_val)
    
        # hyperparameters to optimize
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 1.0) 
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 5e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
        # name run CONFIG!!!
        trial.set_user_attr('run', run_name)
    
        # Neural network configuration
        num_epochs = 10 
        criterion = CrossEntropyLoss() 
        model = FCNN(dropout_rate) # Optimized parameter
        optimizer = Adam(model.parameters(), lr=learning_rate) # Optimized parameter
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Optimized parameter
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Optimized parameter
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # train/validate model
        train_results = train_model(model, criterion, optimizer, num_epochs,
                                    train_loader, val_loader, device, verbose = False)
        # save metrics
        trial.set_user_attr('train_results', train_results)

        # return score
        return train_results['val_accs'][-1]

    # create/load study
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///optuna_DL_exercise.db',  # CONFIG
        study_name='fcn_features1', # CONFIG
        sampler=optuna.samplers.TPESampler(
            n_startup_trials = 10,
            n_ei_candidates = 24,
            seed=42),
        load_if_exists=True  # Continue if study exists
    )

    # set user_attr CONFIG!!!
    study.set_user_attr('script', 'fcn_features1.py')
    study.set_user_attr('dataset', 'poi_dataset.csv')
    study.set_user_attr('model_architecture', 'Custom FCN')
    study.set_user_attr('split_dataset', '80(80train/20val)20test/seed42')
    study.set_user_attr('description', 'Fully-connected neuronal network for described target using described features')
    study.set_user_attr('score', 'accuracy')
    study.set_user_attr('target', 'mean value for MinMaxScaler(Visits) and MinMaxScaler(Likes_Dislikes)')
    study.set_user_attr('numerical_transformer', {
        'type': 'StandardScaler', 
        'columns': ['xps', 'locationLon', 'locationLat', 'NumTags'], 
    })
    study.set_user_attr('categories_transformer', {
        'type': 'MultiLabelBinarizerWrapper', 
        'columns': ['categories'], 
    })
    study.set_user_attr('tier_transformer', {
        'type': 'OneHotEncoder', 
        'columns': ['tier'], 
    })
    study.set_user_attr('numepochs', 10)
    study.set_user_attr('criterion', 'CrossEntropyLoss')
    study.set_user_attr('optimizer', 'Adam')
    
    ## Run trials
    study.optimize(objective, n_trials=30)
    print(f"Completed {len(study.trials)} trials")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    
if __name__ == "__main__":
    main()