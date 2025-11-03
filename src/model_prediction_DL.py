#!/usr/bin/env python3

import torch
import pandas as pd
from torch.utils.data import DataLoader

from utilsModel import TrainModels

def main():
    # TODO Input for this exe
    models_trained = TrainModels()
    print(models_trained.list_of_models)
    config = models_trained.list_of_models['MDL_fcnn_shallow_metadata1_T10']

    print('Reading data...')
    df = pd.read_csv('data/example_dataset.csv')

    print('Preprocessing data...')
    preproc_features = config['preproc_features']
    df_preproc = preproc_features(df)

    print('Processing data...')
    dataset = config['dataset']
    proc = config['proc']
    input_dataset = dataset(df_preproc, **proc)
    dataloader = DataLoader(dataset = input_dataset, shuffle=False, batch_size = 8)

    print('Loading model...')
    architecture = config['model']
    arch_params = config['model_params']
    model = architecture(**arch_params)

    print('Loading parameters...')
    weights_file = config['weights_file']
    model.load_state_dict(torch.load(weights_file, weights_only = True))

    # TODO Use this model as a API (model deployment)
    print('Making predictions...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            for key, value in batch.items():
                batch[key] = value.to(device)
            preds = model(batch)
            predictions.append(preds.cpu())

    predictions = torch.cat(predictions, dim=0)
    probs = torch.softmax(predictions, dim=1)
    classes = torch.argmax(probs, dim=1)
    print(predictions)
    print(probs)
    print(classes)

if __name__ == "__main__":
    main()