#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from utilsClass import TargetFeature, MultiLabelBinarizerWrapper
from utils import process_data, set_random_seed, get_object_info, get_column_transformer_info
from utilsDataset import features_Dataset
from sklearn.ensemble import RandomForestClassifier

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
    proc_features = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), ['xps', 'locationLon', 'locationLat', 'NumTags']),
            ('categories', MultiLabelBinarizerWrapper(), ['categories']), 
            ('tier',OneHotEncoder(sparse_output=False),['tier'])
        ],
        remainder="drop"
    )
    
    proc_target.fit(df_train)
    proc_features.fit(df_train)
    
    # Dataset
    X_train = proc_features.transform(df_train)
    y_train = proc_target.transform(df_train)
    
    ## Configure optimization
    def objective(trial):
        # hyperparameters to optimize
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 25, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0)
        }
        trial.set_user_attr('run', run_name)
        
        # build model
        model = RandomForestClassifier(**params)
        # train/validate model
        cv=5
        cv_results = cross_validate(model, X_train, y_train, cv=cv, return_train_score=True, n_jobs=1)
        score = cv_results['test_score'].mean()
        
        trial.set_user_attr('model', {'name': model.__class__.__name__, 'module': model.__class__.__module__}) 
        trial.set_user_attr('k-fold', cv) 
        trial.set_user_attr('val_score', list(cv_results['test_score'])) 
        trial.set_user_attr('train_score', list(cv_results['train_score'])) 
        
        # return score
        return score
    

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
    study.set_user_attr('proc_features', get_column_transformer_info(proc_features))
    study.set_user_attr('comments', 'Random forest model')
    
    ## Run trials
    study.optimize(objective, n_trials=30)
    print(f"Completed {len(study.trials)} trials")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

if __name__ == "__main__":
    main()