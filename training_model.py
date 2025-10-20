#!/usr/bin/env python3

import optuna
import pandas as pd
from utils import set_random_seed, serial_decode
from sklearn.model_selection import train_test_split
from utilsDataset import features_Dataset
from torch.utils.data import DataLoader

def main():
    # TODO Input storage, studyname, best/number trial
    storage = 'sqlite:///optuna_DL_exercise.db'
    study_name = 'MDL_fcnn_shallow_metadata1'
    study = optuna.load_study(
        study_name = study_name,
        storage = storage
    )

    # Random reproducibility
    set_random_seed()
    random_state = study.user_attrs['random_state']

    # Load dataset
    print('Loading data...')
    data_file = study.user_attrs['datafile']
    print(data_file)
    load = serial_decode(study.user_attrs['load'])
    df = load(data_file) 

    # Proprocess data to train/validate model
    # simple process features + create new features
    print('Preprocessing data...')
    preproc = serial_decode(study.user_attrs['preproc'])
    df_preproc = preproc(df) 
    
    # split data into train and test datasets
    print('Splitting dataset in train and test...')
    test_size_test = study.user_attrs['split_test']
    df_train, df_test = train_test_split(df_preproc, test_size = test_size_test, random_state = random_state, stratify=df_preproc['target'])
    print(f'Train dataset: {df_train.shape[0]}')
    print(f'Test dataset: {df_test.shape[0]}')

    # preprocess data Imputation/Encoding/Transformation
    print('Processing data...')
    proc = serial_decode(study.user_attrs['proc'])
    # TODO Save this proc_features with fit
    proc_to_fit = study.user_attrs['proc_to_fit']
    for proc_fit in proc_to_fit:
        proc[proc_fit].fit(df_train)
    # Dataset
    dataset = serial_decode(study.user_attrs['dataset'])
    train_dataset = dataset(df_train, **proc)
    test_dataset = dataset(df_test, **proc)

    # Training
    print('Loading parameters...')
    trial = study.best_trial
    fixed_params = trial.user_attrs['fixed_params']
    params = trial.user_attrs['params']

    # Build TrainModel
    train_config = serial_decode(study.user_attrs['train_config'])
    model_train = train_config['train']
    train = model_train(train_config, params, fixed_params)

    # Build DataLoader
    train_loader = DataLoader(dataset = train_dataset, shuffle=True, **params['data_loader'], **fixed_params['data_loader']) 
    test_loader = DataLoader(dataset = test_dataset, shuffle=False, **params['data_loader'], **fixed_params['data_loader'])

    # Train model
    print('Training model...')
    num_epochs = trial.user_attrs['num_epochs']
    train.train_model(num_epochs, train_loader)
    # Validate model/Plot
    print('Testing model...')
    train.eval_model(test_loader)
    # TODO Output train/test
    # TODO Output confusion matrix
    # TODO Save model
    # TODO filename???
    # TODO Save info model in trial.user_attrs
    print('Saving model...')
    filename = 'prueba'
    train.save_model(filename)
    # TODO list of dictionaries: study, number trial, filename, score
    trial.set_user_attr('kk', filename)

if __name__ == "__main__":
    main()