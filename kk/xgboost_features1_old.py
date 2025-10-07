#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

from utilsClass import TargetFeature, MultiLabelBinarizerWrapper
from utils import process_data
from utils import set_random_seed

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

    ## Prepare data to model train/validation
    # simple process fetures + create new fetures
    poi_data_processed = process_data(poi_data)
    
    # split data into train and test datasets
    df_train, df_test = train_test_split(poi_data_processed, test_size = 0.2, random_state = 42)
    print(f'Number of samples.')
    print(f'Train dataset: {df_train.shape[0]}')
    print(f'Test dataset: {df_test.shape[0]}')
    
    # preprocess features Imputation/Encoding/Transformation (target + explanatory)
    preproc_target = TargetFeature(col1_name='Visits', col2_name='Likes_Dislikes')
    preproc_explanatory = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), ['xps', 'locationLon', 'locationLat', 'NumTags']),
            ('categories', MultiLabelBinarizerWrapper(), ['categories']), 
            ('tier',OneHotEncoder(sparse_output=False),['tier'])
        ],
        remainder="drop"
    )
    preproc_union = FeatureUnion([
        ('explanatory', preproc_explanatory),
        ('target', preproc_target)
    ])
    
    X_train = preproc_explanatory.fit_transform(df_train)
    y_train = preproc_target.fit_transform(df_train)
    X_test = preproc_explanatory.transform(df_test)
    y_test = preproc_target.transform(df_test)

    ## Configure optimization
    def objective(trial):
        # hyperparams to optimize
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'n_estimators': trial.suggest_int('n_estimators', 25, 700),
            'eta': trial.suggest_float('eta', 1e-3,1, log = True),
            'reg_alpha': trial.suggest_int('reg_alpha', 6, 10),
            'reg_lambda': trial.suggest_int('reg_lambda', 0, 20)
        }
        # name run CONFIG!!!
        trial.set_user_attr('run',run_name)
            
        # build model
        model = XGBClassifier(**params, seed=42)
        # train/validate model
        score = cross_val_score(model, X_train, y_train, cv=5, n_jobs=1).mean()
        # return score
        return score

    # create/load study
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///optuna_DL_exercise.db',  # CONFIG
        study_name='xgboost_features1', # CONFIG
        sampler=optuna.samplers.TPESampler(
            n_startup_trials = 10,
            n_ei_candidates = 24,
            seed=42),
        load_if_exists=True  # Continue if study exists
    )

    # set user_attr CONFIG!!!
    study.set_user_attr('script', 'xgboost_features1.py')
    study.set_user_attr('dataset', 'poi_dataset.csv')
    study.set_user_attr('model_architecture', 'XGBClassifier')
    study.set_user_attr('split_dataset', 'train80/test20/seed42')
    study.set_user_attr('description', 'XGBClassifier for described target using described features')
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
    
    ## Run trials
    study.optimize(objective, n_trials=10)
    print(f"Completed {len(study.trials)} trials")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

if __name__ == "__main__":
    main()