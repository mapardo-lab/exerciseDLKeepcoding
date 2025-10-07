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

from utilsClass import TargetFeature, MultiLabelBinarizerWrapper
from sklearn.compose import ColumnTransformer
from utilsClass import TargetFeature
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils import process_data, set_random_seed
from utilsDataset import img_meta_Dataset
from utilsNN import dual_ResNet18_layer4
#from utilsTrain import train_model
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
    poi_data = pd.read_csv("poi_dataset.csv")

    ## Prepare data to train/validate model
    # simple process features + create new features
    poi_data_processed = process_data(poi_data)
    
    # split data into train and test datasets
    df_train, df_test = train_test_split(poi_data_processed, test_size = 0.2, random_state = 42)
    print(f'Number of samples.')
    print(f'Train dataset: {df_train.shape[0]}')
    print(f'Test dataset: {df_test.shape[0]}')
    
    # preprocess data Imputation/Encoding/Transformation
    preproc_target = TargetFeature(col1_name='Visits', col2_name='Likes_Dislikes')
    preproc_explanatory = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), ['xps', 'locationLon', 'locationLat', 'NumTags']),
            ('categories', MultiLabelBinarizerWrapper(), ['categories']), 
            ('tier',OneHotEncoder(sparse_output=False),['tier'])
        ],
        remainder="drop"
    )

    
    preproc_explanatory.fit(df_train)
    preproc_target.fit(df_train)
    
    #X_train = preproc_explanatory.fit_transform(df_train)
    #y_train = preproc_target.fit_transform(df_train)
    #X_test = preproc_explanatory.transform(df_test)
    #y_test = preproc_target.transform(df_test)
    
    #X_train = np.array(df_train['main_image_path'])
    #y_train = preproc_target.fit_transform(df_train)
    #X_test = np.array(df_test['main_image_path'])
    #y_test = preproc_target.transform(df_test)
    
    # split train data into train and validation datasets
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
    df_train, df_val = train_test_split(df_train, test_size = 0.2, random_state = 42)

    transform_ResNet18 = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    #train_dataset = img_meta_Dataset(y_train, X_train, transform_img = transform_ResNet18)
    #val_dataset = img_meta_Dataset(y_val, X_val, transform_img = transform_ResNet18)
    train_dataset = img_meta_Dataset(df_train, transform_img = transform_ResNet18, transform_meta = preproc_explanatory, transform_target = preproc_target)
    val_dataset = img_meta_Dataset(df_val, transform_img = transform_ResNet18, transform_meta = preproc_explanatory, transform_target = preproc_target)
    
    ## Configure optimization
    def objective(trial):
    
        # hyperparameters to optimize
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 1.0) 
        learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-1, log=True) # To optimize
#        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]) # To optimize
        batch_size = 2**trial.suggest_int("batch_size_exp2", 4, 9) # To optimize
        # name run CONFIG!!!
        trial.set_user_attr('run', run_name)
    
        # Neural network configuration
        num_epochs = 10 
        criterion = CrossEntropyLoss() 
        model = dual_ResNet18_layer4(dropout_rate) # Optimized parameter
        optimizer = Adam(model.parameters(), lr=learning_rate) # Optimized parameter
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Optimized parameter
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Optimized parameter
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_results = Train()
        model.to(device)

        #train_losses, train_accs, val_losses, val_accs = [], [], [], []
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
        study_name='resnet18_layer4_img1_fea1',
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
    study.set_user_attr('script', 'resnet18_layer4_img1_fea1.py')
    study.set_user_attr('dataset', 'poi_dataset.csv')
    study.set_user_attr('model_architecture', 'Ensemble model: ResNet18 pretrained optimization for layer4 (images) + FCNN (metadata)')
    study.set_user_attr('split_dataset', '80(80train/20val)20test/seed42')
    study.set_user_attr('description', 'Emsemble model with multi-modal input. Images: ResNet18 with optimization for layer4. Metadata: FCNN. Feature concatenation ')
    study.set_user_attr('score', 'accuracy')
    study.set_user_attr('target', 'mean value for MinMaxScaler(Visits) and MinMaxScaler(Likes_Dislikes)')
    study.set_user_attr('img_transformation', 'ResNet transformation')
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
    study.optimize(objective, n_trials=10)
    print(f"Completed {len(study.trials)} trials")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

if __name__ == "__main__":
    main()