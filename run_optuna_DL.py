#!/usr/bin/env python3

import sys
import optuna
import torch
from sklearn.model_selection import train_test_split
from utils import set_random_seed, save_objects, load_yaml_config, load_from_config, load_dict_from_config
from utilsOptuna import ObjectiveFunctionDL, create_study

def main():
    # Check if run_name argument is provided
    if len(sys.argv) < 2:
        print("Error: No run name provided")
        print("Usage: script.py <.yaml file>")
        sys.exit(1)
    print('Reading configuration .yaml file...')
    config = load_yaml_config(sys.argv[1])

    study_name = sys.argv[1].replace('.yaml','').replace('./','')
    run_name = config['run_name']
    print(f'Study: {study_name} Run: {run_name}')    
    
    print('Setting random state for reproducibility...')
    set_random_seed()
    random_state = config['random_state']
    
    print('Loading dataset...')
    data_file = config['data_file']
    load = load_from_config(config['load_data'])
    df = load(data_file) 

    print('Preprocessing data... simple processing + new features')
    # TODO As transform metadata (class with function)
    preproc_features = load_from_config(config['preproc_features'])
    preproc_target = load_from_config(config['preproc_target'])
    df_processed = preproc_target(preproc_features(df))
    
    # split data into train and test datasets
    print('Splitting dataset in train and test...')
    test_size_test = config['test_size_test']
    df_train, df_test = train_test_split(df_processed, test_size = test_size_test, random_state = random_state, stratify=df_processed['target']) 
    print(f'Train: {df_train.shape[0]}/Test: {df_test.shape[0]}')
    
    # preprocess data Imputation/Encoding/Transformation
    print('Loading transformation data process...')
    proc = load_dict_from_config(config['proc'])
    proc_to_fit = config['proc_to_fit']
    for proc_fit in proc_to_fit:
        proc[proc_fit].fit(df_train)

    # split train data into train and validation datasets
    print('Splitting dataset into training and validation dataset...')
    test_size_val = config['test_size_val']
    df_train, df_val = train_test_split(df_train, test_size = test_size_val, random_state = 42, stratify = df_train['target']) 

    # Dataset
    print('Loading datasets modules...')
    dataset = load_from_config(config['dataset'])
    train_dataset = dataset(df_train, **proc)
    val_dataset = dataset(df_val, **proc)
    
    # hyperparameter to optimizate
    print('Loadind search space...')
    search_space = config['search_space']

    # fixed hyperparameters
    print('Loading fixed parameters...')
    fixed_params = config['fixed_params']

    # scores output
    print('Loading scores to monitorize...')
    scoring = load_dict_from_config(config['scoring'])

    # model configuration 
    print('Loadind model configuration...')
    model_config = load_dict_from_config(config['model_config'])
    model_config['scoring'] = scoring
    if config['device'] == 'gpu':
        model_config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        model_config['device'] = 'cpu'

    # objetive function DL
    print('Preparing hyperparameter optimization...')
    objective = ObjectiveFunctionDL(
        train_dataset = train_dataset, val_dataset = val_dataset,
        model_config = model_config,
        fixed_params=fixed_params,
        search_space=search_space,
        score = config['score'],
        run_name=run_name
    )

    # parameters for study
    # TODO Write parameters in .yaml file
    study_params = {
        'direction': 'maximize',
        'storage': 'sqlite:///optuna_DL_exercise.db',  # Persistent storage
        'study_name': study_name,
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

    print('Saving configuration...')
    save_config = {
        'random_state': random_state,
        'datafile': data_file,
        'load': load,
        'preproc_features': preproc_features,
        'preproc_target': preproc_target,
        'split_test': test_size_test,
        'proc': proc,
        'proc_to_fit': proc_to_fit,
        'dataset': dataset,
        'split_val': test_size_val,
        'model_config': model_config
    }
    file_config = study_name + '.pkl'
    save_objects(save_config, file_config)

    # user attributes for study
    user_attr = [
        ('script', f'{sys.argv[0]}'),
        ('config_file', file_config),
        ('comments', config['comments'])
    ]

    # create study
    study = create_study(study_params, user_attr)

    ## run trials
    print('Running hyperparameter optimization...')
    study.optimize(objective, n_trials=config['num_trials'])
    print(f"Completed {len(study.trials)} trials")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

if __name__ == "__main__":
    main()