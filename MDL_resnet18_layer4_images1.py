#!/usr/bin/env python3

import sys
import pandas as pd
import optuna
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import recall_score, precision_score, f1_score

from utilsFT import ImagesResNet18Transform
from utils import set_random_seed, build_scorer, serial_encode
from utilsPreproc import preprocess_data 
from utilsDataset import images_Dataset
from utilsOptuna import ObjectiveFunctionDL, create_study
from utilsTrain import ModelTrain
from utilsNN import ResNet18_layer4

def main():
    ## Check if run_name argument is provided
    if len(sys.argv) < 2:
        print("Error: No run name provided")
        print("Usage: script.py <run_name>")
        sys.exit(1)
    run_name = sys.argv[1]
    
    ## Random reproducibility
    set_random_seed()
    random_state = 42
    
    ## Load dataset
    data_file = "poi_dataset.csv"
    load = pd.read_csv
    df = load(data_file) 

    ## Proprocess data to train/validate model
    # simple process features + create new features
    preproc = preprocess_data
    df_processed = preproc(df) 
    
    # split data into train and test datasets
    test_size_test = 0.2
    df_train, df_test = train_test_split(df_processed, test_size = test_size_test, random_state = random_state, stratify=df_processed['target']) 
    print(f'Number of samples')
    print(f'Train dataset: {df_train.shape[0]}')
    print(f'Test dataset: {df_test.shape[0]}')
    
    # preprocess data Imputation/Encoding/Transformation
    proc = {
        'transform_images': ImagesResNet18Transform(image_path='main_image_path')
    }
    proc_to_fit = []

    # split train data into train and validation datasets
    test_size_val = 0.2
    df_train, df_val = train_test_split(df_train, test_size = test_size_val, random_state = 42, stratify = df_train['target']) 

    # Dataset
    dataset = images_Dataset
    train_dataset = dataset(df_train, **proc)
    val_dataset = dataset(df_val, **proc)
    
    # hyperparameter to optimizate
    search_space = {
        'data_loader': {
            'batch_size': {'type': 'int', 'low': 2, 'high': 5, 'exp2': True},
        },
        'model': {},
        'optimizer': {
            'lr': {'type': 'float', 'low': 5e-4, 'high': 5e-1, 'log': True},
        },
        'criterion': {}
    }

    # fixed hyperparameters
    fixed_params = {
        'data_loader': {},
        'model': {
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

    train_config = {
        'num_epochs': 10,
        'train': ModelTrain,
        'model': ResNet18_layer4,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
        'criterion': CrossEntropyLoss,
        'optimizer': Adam,
        'scoring': scoring
    }

    # objetive function DL
    objective = ObjectiveFunctionDL(
        train_dataset = train_dataset, val_dataset = val_dataset,
        train_config = train_config,
        fixed_params=fixed_params,
        search_space=search_space,
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
        ('random_state', random_state),
        ('datafile', f'{data_file}'),
        ('load', serial_encode(load)),
        ('preproc', serial_encode(preproc)),
        ('split_test', test_size_test),
        ('proc', serial_encode(proc)),
        ('proc_to_fit', proc_to_fit),
        ('dataset', serial_encode(dataset)),
        ('split_val', test_size_val),
        ('train_config', serial_encode(train_config)),
        ('comments', 'ResNet18 layer4 parameters optimization')
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