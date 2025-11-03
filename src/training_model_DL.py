#!/usr/bin/env python3

import optuna
import sys
import os
from datetime import datetime
from utils import set_random_seed
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import load_objects, load_dict_from_config
from utilsModel import TrainModels

def main():
    optuna_dir = 'optuna'
    pkl_dir = os.path.join(optuna_dir,'file_config')
    # TODO Select best trial or number of trial
    # Check if study_name argument is provided
    if len(sys.argv) < 2:
        print("Error: No study name provided")
        print("Usage: script.py <study_name>")
        sys.exit(1)
    study_name = sys.argv[1]

    storage = 'sqlite:///optuna/optuna_DL_exercise.db'
    study = optuna.load_study(
        study_name = study_name,
        storage = storage
    )
    trial = study.best_trial
    trial_number = trial.number
    config = load_objects(os.path.join(pkl_dir, study_name + '.pkl'))

    # Random reproducibility
    set_random_seed()
    random_state = config['random_state']

    # Load dataset
    print('Loading data...')
    data_file = config['datafile']
    load = config['load']
    df = load(data_file) 

    # Proprocess data to train/validate model
    # simple process features + create new features
    print('Preprocessing data...')
    preproc_features = config['preproc_features']
    preproc_target = config['preproc_target']
    df_preproc = preproc_features(preproc_target(df))
    
    # split data into train and test datasets
    print('Splitting dataset in train and test...')
    test_size_test = config['split_test']
    df_train, df_test = train_test_split(df_preproc, test_size = test_size_test, random_state = random_state, stratify=df_preproc['target'])
    print(f'Train: {df_train.shape[0]}/Test: {df_test.shape[0]}')

    # preprocess data Imputation/Encoding/Transformation
    print('Processing data...')
    proc = load_dict_from_config(config['proc'])
    proc_to_fit = config['proc_to_fit']
    for proc_fit in proc_to_fit:
        proc[proc_fit].fit(df_train)
    # Dataset
    dataset = config['dataset']
    train_dataset = dataset(df_train, **proc)
    test_dataset = dataset(df_test, **proc)

    # Training
    print('Loading parameters...')
    fixed_params = trial.user_attrs['fixed_params']
    params = trial.user_attrs['params']

    # Build Model
    model_config = config['model_config']
    model_train = model_config['train']
    train = model_train(model_config, params, fixed_params)

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
    print('Saving model...')
    model_name = study_name + '_T' + str(trial_number)
    file_weights = os.path.join('models', model_name)
    train.save_model(file_weights)
    print('Writing model info...')
    models_trained = TrainModels()
    # TODO Save scores for test
    info_model = {
        'study': study_name,
        'number_trial': trial_number,
        'data_file': data_file,
        'preproc_features': preproc_features,
        'proc': proc,
        'dataset': dataset,
        'weights_file': file_weights + '.pth',
        'model': model_config['architecture'],
        'model_params': params['architecture'] | fixed_params['architecture'],
        'date': datetime.now(),
        'scores': train.results_val.results
    }
    models_trained.append_model(model_name, info_model)
    models_trained.save()

if __name__ == "__main__":
    main()