#!/usr/bin/env python3

import torch
import optuna
from utils import serial_decode

def main():
    # TODO First I have to design a way to save info about all models that have been trained (study, trial, scores, filename weights)
    # TODO Give to this exe file info about the model I want to run
    storage = 'sqlite:///optuna_DL_exercise.db'
    study_name = 'fcnn_shallow_metadata1'
    study = optuna.load_study(
        study_name = study_name,
        storage = storage
    )

    print('Loading parameters...')
    trial = study.best_trial
    fixed_params = trial.user_attrs['fixed_params']
    params = trial.user_attrs['params']
    train_config = serial_decode(study.user_attrs['train_config'])

    print('Loading model...')
    architecture = train_config['model']
    arch_params = params['model'] | fixed_params['model']
    model = architecture(**arch_params)

    model.load_state_dict(torch.load("prueba.pth", weights_only = True))
    model.eval()  # set to evaluation mode before inference
    # TODO Use this model as a API (model deployment)

    # TODO Preprocess + Process data
    #with torch.no_grad():
    #    x_new = torch.randn(5, 10)
    #    predictions = model(x_new)
    #print(predictions)

if __name__ == "__main__":
    main()