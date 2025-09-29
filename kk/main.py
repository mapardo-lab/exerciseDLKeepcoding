#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from optuna.samplers import RandomSampler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from utils import read_images, get_scaler_engagement, get_score_engagement, categorize_engagement_score, get_normalize_images
from utilsOptuna import optuna_init
from utilsDataset import img_Dataset
from utilsNN import CNN
from utilsTrain import train_model

def processdata(df):
    # This could be copy into Dataset. And make different Datasets with different names????
    """
    Data processing steps before being used in the model.
    Same steps are applied to train, validation and test datasets.
    Processing include:
    - Calculate engagement feature (low, medium, high)
    """
    df['score'] = get_score_engagement(df, scaler_engagement)
    df['engagement'] = df['score'].apply(categorize_engagement_score)
    return df

if __name__ == "__main__":
    # load dataset
    poi_data = pd.read_csv("poi_dataset.csv")

    # split data into train, val and test datasets
    df_train, df_test = train_test_split(poi_data, test_size = 0.2, random_state = 42)
    df_train, df_val = train_test_split(df_train, test_size = 0.2, random_state = 42)

    # Preprocessing
    # Calculate mean and std for each channel of images from train dataset (This will be used for normalization after transform to tensor)
    # TODO This could be save in a dictionary
    images = read_images(df_train['main_image_path'])
    normalize_images = get_normalize_images(images)
    # get MinMaxScaler for engagement rows (This will be used for engagement calculation in different dataset
    scaler_engagement = get_scaler_engagement(df_train)

    df_train_proc = processdata(df_train)
    df_val_proc = processdata(df_val)
    df_test_proc = processdata(df_test)

    # train model (check if configuration is ok)
    # Prepare Datasets (train and val)
    transform_img_norm = transforms.Compose([ # config
        transforms.ToTensor(),
        #transforms.Normalize(means, stds)
        transforms.Normalize(normalize_images[0], normalize_images[1])
    ])
    train_dataset = img_Dataset(df_train_proc['engagement'], df_train_proc['main_image_path'], transform_img = transform_img_norm)
    val_dataset = img_Dataset(df_val_proc['engagement'], df_val_proc['main_image_path'], transform_img = transform_img_norm)
    #set_random_seed()
    learning_rate = 0.01 # To optimize
    dropout_rate = 0.2 # To optimize
    batch_size = 128 # To optimize
    num_epochs = 20 
    criterion = CrossEntropyLoss() 
    model = CNN(dropout_rate) # Optimized parameter
    optimizer = Adam(model.parameters(), lr=learning_rate) # Optimized parameter
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Optimized parameter
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Optimized parameter
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    train_model(model, criterion, optimizer, num_epochs, train_loader, val_loader, device)
