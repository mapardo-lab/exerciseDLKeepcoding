#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.compose import ColumnTransformer
from utilsFT import TargetFeature, EmbeddingText
from utils import set_random_seed, get_object_info, get_column_transformer_info
from utilsProc import process_data
from utilsDataset import features_Dataset
from utilsNN import FCNN_pca
from utilsTrain import Train, train_epoch, eval_epoch

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
    poi_data = pd.read_csv(data_file) # *

    ## Proprocess data to train/validate model
    # simple process features + create new features
    poi_data_processed = process_data(poi_data) # *
    
    # split data into train and test datasets
    test_size_test = 0.2
    df_train, df_test = train_test_split(poi_data_processed, test_size = test_size_test) # *
    print(f'Number of samples')
    print(f'Train dataset: {df_train.shape[0]}')
    print(f'Test dataset: {df_test.shape[0]}')
    
    # preprocess data Imputation/Encoding/Transformation
    proc_target = TargetFeature(col1_name='Visits', col2_name='Likes_Dislikes') # *
    proc_features = EmbeddingText('shortDescription', 'all-MiniLM-L6-v2') # *
    
    proc_target.fit(df_train)
    
    # split train data into train and validation datasets
    test_size_val = 0.2
    df_train, df_val = train_test_split(df_train, test_size = test_size_val) # *

    # Dataset
    train_dataset = features_Dataset(df_train, transform_features = proc_features, transform_target = proc_target)
    val_dataset = features_Dataset(df_val, transform_features = proc_features, transform_target = proc_target)
    
    ## Configure optimization
    def objective(trial):
    
        # hyperparameters to optimize
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 1.0) 
        learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-1, log=True) # To optimize
        batch_size = 2**trial.suggest_int("batch_size_exp2", 2, 9) # To optimize
        # name run CONFIG!!!
        trial.set_user_attr('run', run_name)
    
        # Neural network configuration
        num_epochs = 10 # *
        criterion = CrossEntropyLoss() # *
        model = FCNN_pca(384, num_classes = 2, dropout_rate = dropout_rate) # Optimized parameter dropout_rate
        optimizer = Adam(model.parameters(), lr=learning_rate) # Optimized parameter learning_rate
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Optimized parameter batch_size
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Optimized parameter batch_size
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # *
        trial.set_user_attr('num_epochs', num_epochs)
        trial.set_user_attr('criterion', f'{criterion}')
        trial.set_user_attr('model', {'name': model.__class__.__name__, 'module': model.__class__.__module__}) 
        trial.set_user_attr('optimizer', f'{Adam}')
        trial.set_user_attr('device', f'{device}') 

        train_results = Train()
        model.to(device)

        for epoch in range(num_epochs):
            loss, acc , lr = train_epoch(model, device, train_loader, criterion, optimizer)
            val_loss, val_acc = eval_epoch(model, device, val_loader, criterion)
            train_results.update(loss, acc, val_loss, val_acc)

            # Report to pruner
            trial.report(val_acc, step = epoch)

            # Check for pruning
            if trial.should_prune():
                # save metrics
                trial.set_user_attr('train_results', train_results.to_dict())
                print(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()
        
        # save metrics
        final_train_results = train_results.to_dict()
        trial.set_user_attr('train_results', final_train_results)
    
        # return score
        return final_train_results['val_accs'][-1]

    # create/load study
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///optuna_DL_exercise.db',  # Persistent storage
        study_name=sys.argv[0].replace('.py','').replace('./',''),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials = 10,
            n_ei_candidates = 24,
            seed=42),
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 5,
            n_warmup_steps = 3
        ),
        load_if_exists=True  # Continue if study exists
    )

    # set user_attr CONFIG!!!
    study.set_user_attr('script', f'{sys.argv[0]}')
    study.set_user_attr('dataset', f'{data_file}')
    study.set_user_attr('preproc_data', {'module': process_data.__module__, 'function': process_data.__name__})
    study.set_user_attr('split_test', {'test_size': test_size_test})
    study.set_user_attr('proc_target', get_object_info(proc_target))
    study.set_user_attr('proc_features', get_object_info(proc_features))
    study.set_user_attr('split_val', {'test_size': test_size_val})
    study.set_user_attr('comments', 'FCNN with Progressive Compression Architecture for embeddings')
    
    ## Run trials
    study.optimize(objective, n_trials=40)
    print(f"Completed {len(study.trials)} trials")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

if __name__ == "__main__":
    main()