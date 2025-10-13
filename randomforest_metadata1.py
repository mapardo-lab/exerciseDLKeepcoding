#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score

from utilsFT import MultiLabelBinarizerWrapper
from utils import set_random_seed, get_column_transformer_info
from utilsProc import process_data
from utilsOptuna import ObjectiveFunctionML, create_study

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
    df_train, df_test = train_test_split(poi_data_processed, test_size = test_size_test, stratify = poi_data_processed['target'], random_state=42) # *
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
    
    # Dataset
    X_train = proc_features.transform(df_train)
    y_train = df_train['target']

    # hyperparameters to optimizate
    search_space = {
        'model': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'max_features': {'type': 'float', 'low': 0.1, 'high': 1.0},
        }
    }

    # fixed hyperparameters
    fixed_params = {
        'model': {
            'random_state': 42
        }
    }

    scoring = {
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'precision': make_scorer(precision_score, pos_label=1),
        'f1_score': make_scorer(f1_score, pos_label=1),
        'macro_precision': make_scorer(precision_score, average='binary')
    }

    # objetive function ML
    objective = ObjectiveFunctionML(
        X=X_train, y=y_train,
        model=RandomForestClassifier,
        fixed_params=fixed_params,
        search_space=search_space,
        scoring=scoring,
        score = 'f1_score',
        cv = 5,
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
        'pruner': None, 
        'load_if_exists': True  # Continue if study exists
    }

    # user attributes for study
    user_attr = [
        ('script', f'{sys.argv[0]}'),
        ('dataset', f'{data_file}'),
        ('preproc_data', f'{type(process_data)}'),
        ('split_test', {'test_size': test_size_test}),
        ('proc_features', get_column_transformer_info(proc_features)),
        ('comments', 'RandomForest')
    ]

    # create study
    study = create_study(study_params, user_attr)

    ## run trials
    study.optimize(objective, n_trials=20)
    print(f"Completed {len(study.trials)} trials")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

if __name__ == "__main__":
    main()