import torch
import numpy  as np  
import pandas as pd
import torch.nn as nn
from torchvision import models

class FCNN(nn.Module):
  """
  Fully Connected Neural Network (FCNN) with an 
  - 20 neurons input layer  hidden layer with 
  - 32 neuron hidden layer (ReLU activation function and dropout regularization)
  - 3 neurons output layer
  """
  def __init__(self, dropout_rate):
    super(FCNN, self).__init__()
    # First fully connected layer
    self.layer1 = nn.Sequential(
      nn.Linear(20, 32),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Dropout(dropout_rate)
    )

    # Output fully connected layer
    self.layer2 = nn.Sequential(
      nn.Linear(32, 3)
    )

  def forward(self, data):
    x = data['meta'] 
    x = self.layer1(x)
    x = self.layer2(x)
    return x


class CNN(nn.Module):
  """
  Convolutional Neural Network (CNN) with an 
  - Convolutional layer for three input channels with 8 kernels (3x3) and padding 1.
  Use batch normalization before apply a ReLU activation function. Max pooling and 
  dropout are applied
  - Global max and average pooling is applied
  - 8 neuron fully-connected layer for the 16 input features from above steps
  Batch normalization, ReLU activation function and dropout regularization are applied
  - 3 neurons output layer
  """
  def __init__(self, dropout_rate):
    super(CNN, self).__init__()

    # First convolutional layer
    self.convLayer1 = nn.Sequential(
      nn.Conv2d(3, 8, 3, padding = 1),
      nn.BatchNorm2d(8),
      nn.ReLU(),
      nn.MaxPool2d(2,2),
      nn.Dropout(dropout_rate)
    )

    # Fully connected layer (classificator)
    self.fcLayer1 = nn.Sequential(
      nn.Linear(16, 8),
      nn.BatchNorm1d(8),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(8, 3)
    )

    # Set global pooling (max/avg)
    self.global_max_pool = nn.AdaptiveMaxPool2d(1) # torch.nn.AdaptiveMaxPool2d(output_size,...)
    self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

  def forward(self, data):
    x = data['img'] 
    x = self.convLayer1(x)
    max_pooled = self.global_max_pool(x).squeeze()
    avg_pooled = self.global_avg_pool(x).squeeze()
    x = torch.cat((max_pooled, avg_pooled), dim=1)
    x = self.fcLayer1(x)
    return x 

class dual_branch(nn.Module):
  """
  Dual-branch neuronal network
  """
  def __init__(self, dropout_rate):
    super(dual_branch, self).__init__()

    # CNN: First convolutional layer
    self.CNN_convLayer1 = nn.Sequential(
      nn.Conv2d(3, 8, 3, padding = 1),
      nn.BatchNorm2d(8),
      nn.ReLU(),
      nn.MaxPool2d(2,2),
      nn.Dropout(dropout_rate)
    )

    # CNN: Set global pooling (max/avg)
    self.global_max_pool = nn.AdaptiveMaxPool2d(1) # torch.nn.AdaptiveMaxPool2d(output_size,...)
    self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    # FCNN: First fully connected layer
    self.FCNN_fcLayer1 = nn.Sequential(
      nn.Linear(20, 16),
      nn.BatchNorm1d(16),
      nn.ReLU(),
      nn.Dropout(dropout_rate)
    )

    # Classificator: Fully connected layer
    self.class_fcLayer1 = nn.Sequential(
      nn.Linear(32, 16),
      nn.BatchNorm1d(16),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(16, 3)
    )

  def forward(self, data):
    x_cnn = data['img']
    x_fcnn = data['meta']
    x_cnn = self.CNN_convLayer1(x_cnn)
    max_pooled = self.global_max_pool(x_cnn).squeeze()
    avg_pooled = self.global_avg_pool(x_cnn).squeeze()
    x_cnn = torch.cat((max_pooled, avg_pooled), dim=1)
    x_fcnn = self.FCNN_fcLayer1(x_fcnn)
    x = torch.cat((x_cnn, x_fcnn), dim = 1)
    x = self.class_fcLayer1(x)
    return x

def ResNet18_class():
  """
  Initializes a pretrained ResNet18 model for transfer learning with the following modifications:
  1. Loads weights pretrained on ImageNet
  2. Freezes all layers to prevent gradient updates
  3. Replaces the final fully-connected layer for binary classification
  """
  # build a model based on ResNet18
  model = models.resnet18(pretrained=True)
  model.eval()  # Set to evaluation mode

  # freeze all layers of ResNet18 model so they are not trained (transfer learning)
  for param in model.parameters():
    param.requires_grad = False

  # Change classificator by smaller one
  num_features = model.fc.in_features # input feature to classificator
  model.fc = torch.nn.Linear(num_features, 2) # Three levels engagement

  return model

def ResNet18_branch():
  """
  Creates a modified ResNet18 feature extractor branch for transfer learning by:
  1. Loading a pretrained ResNet18 model (ImageNet weights)
  2. Freezing all layers to prevent training
  3. Removing the final classification layer (keeping only feature extraction layers)
  """
  # build a model based on ResNet18
  model = models.resnet18(pretrained=True)
  model.eval()  # Set to evaluation mode

  # freeze all layers of ResNet18 model so they are not trained (transfer learning)
  for param in model.parameters():
    param.requires_grad = False

  # Remove classificator layer
  model = torch.nn.Sequential(*list(model.children())[:-1])

  return model

class dual_branch_ResNet18(nn.Module):
  def __init__(self, dropout_rate):
    super(dual_branch_ResNet18, self).__init__()

    # CNN branch: ResNet18 model
    self.resnet18 = ResNet18_branch()

    # FCNN: First fully connected layer
    self.FCNN_fcLayer1 = nn.Sequential(
      nn.Linear(20, 16),
      nn.BatchNorm1d(16),
      nn.ReLU(),
      nn.Dropout(dropout_rate)
    )

    # Classificator: Fully connected layer
    self.class_fcLayer1 = nn.Sequential(
      nn.Linear(528, 16),
      nn.BatchNorm1d(16),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(16, 2)
    )

  def forward(self, data):
    x_cnn = data['img']
    x_resnet18 = self.resnet18(x_cnn).squeeze()
    x_fcnn = data['meta']
    x_fcnn = self.FCNN_fcLayer1(x_fcnn)
    x = torch.cat((x_resnet18, x_fcnn), dim = 1)
    x = self.class_fcLayer1(x)
    return x

class CNN_pretrain(nn.Module):
  """
  Convolutional Neural Network (CNN) using a pretrained model
  """
  def __init__(self, model):
    super(CNN_pretrain, self).__init__()
    self.model = model()

  def forward(self, data):
    x = data['img'] 
    x = self.model(x)
    return x 