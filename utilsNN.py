import torch
import numpy  as np  
import pandas as pd
import torch.nn as nn
from torchvision import models
import torch.nn as nn

class CommonBlocks:
    """
    Container for reusable sequential blocks
    """
    
    @staticmethod
    def get_fcnn_shallow_classifier(input_size, output_size=2, dropout=0.3):
        """Fully Connected Neural Network (FCNN). Shallow architecture."""
        return nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(16, output_size),
        )
        
    @staticmethod
    def get_fcnn_pca_classifier(input_size, output_size=2, dropout=0.3):
        """Fully Connected Neural Network (FCNN) with progressive compression architecture"""
        return nn.Sequential(
            # layer1: Reduce dimensionality
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
                
            # layer2: Futher compression
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
                
            # layer3: Final features
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # output layer
            nn.Linear(64, output_size),
        )

    @staticmethod
    def get_ResNet18_pre_classifier(output_size=2): #
      """
      Initializes a pretrained ResNet18 model for transfer learning with the following modifications:
      1. Loads weights pretrained on ImageNet
      2. Freezes all layers to prevent gradient updates
      3. Replaces the final fully-connected layer for binary classification
      """
      # build a model based on ResNet18
      model = models.resnet18(weights='IMAGENET1K_V1')
      #model.eval()  # Set to evaluation mode
    
      # freeze all layers of ResNet18 model so they are not trained (transfer learning)
      for param in model.parameters():
        param.requires_grad = False
    
      # Change classificator by smaller one
      num_features = model.fc.in_features # input feature to classificator
      model.fc = torch.nn.Linear(num_features, output_size) # Two levels engagement
    
      return model
        
    @staticmethod
    def get_ResNet18_layer4_classifier(output_size=2): #
        """
        Initializes a pretrained ResNet18 model for transfer learning with the following modifications:
        1. Loads weights pretrained on ImageNet
        2. Freezes all layers except layer4 
        3. Replaces the final fully-connected layer for binary classification
        """
        # build a model based on ResNet18
        model = models.resnet18(weights='IMAGENET1K_V1')
        #model.eval()  # Set to evaluation mode
    
        # Freeze early layers, keep later layers trainable
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:  # Freeze everything except layer4 and fc
                param.requires_grad = False
            else:
                param.requires_grad = True
    
        # Change classificator by smaller one
        num_features = model.fc.in_features # input feature to classificator
        model.fc = torch.nn.Linear(num_features, output_size) # Two levels engagement
        
        return model
        
    
class FCNN_shallow(nn.Module): # 
    """
    Fully Connected Neural Network (FCNN). Shallow architecture.
    """
    def __init__(self, input_size, num_classes=2, dropout_rate = 0.4):
        super(FCNN_shallow, self).__init__()
        self.network = CommonBlocks.get_fcnn_shallow_classifier(input_size, num_classes, dropout_rate)

    def forward(self, data):
        x = data['features'] 
        x = self.network(x)
        return x
        
class FCNN_pca(nn.Module): #
    """
    Fully Connected Neural Network (FCNN) with progressive compression architecture
    """
    def __init__(self, input_size, num_classes = 2, dropout_rate = 0.4):
        super(FCNN_pca, self).__init__()
        self.network = CommonBlocks.get_fcnn_pca_classifier(input_size, num_classes, dropout_rate)

    def forward(self, data):
        x = data['features'] 
        x = self.network(x)
        return x
        
class CNN_pretrain(nn.Module): #
  """
  Convolutional Neural Network (CNN) using a pretrained model
  """
  def __init__(self, model):
    super(CNN_pretrain, self).__init__()
    self.model = model

  def forward(self, data):
    x = data['images'] 
    x = self.model(x)
    return x 
      
def last_layer_input(model):
        last_layer = list(model.children())[-1]
        return last_layer.in_features

class multi_modal3_class2(nn.Module):
    def __init__(self, metadata_input, embeddings_input, dropout_rate):
        super(multi_modal3_class2, self).__init__()
        
        # Images branch
        self.branch_images = CommonBlocks.get_ResNet18_layer4_classifier()
        images_size = last_layer_input(self.branch_images)        
        self.branch_images.fc = nn.Identity()
        
        # Embeddings branch
        fcnn_pca_classifier = CommonBlocks.get_fcnn_pca_classifier(embeddings_input, dropout=dropout_rate)
        embeddings_size = last_layer_input(fcnn_pca_classifier)        
        self.branch_embeddings = nn.Sequential(*list(fcnn_pca_classifier.children())[:-1])
        
        total_features = images_size + embeddings_size + metadata_input
    
        # Classificator: Fully connected layer
        self.classifier = CommonBlocks.get_fcnn_pca_classifier(total_features, 2, dropout_rate)
    
    def forward(self, data):
        x_images = data['images']
        #x_images = self.model_images(x_images).squeeze()
        x_images = self.branch_images(x_images)
        x_embeddings = data['embeddings']
        x_embeddings = self.branch_embeddings(x_embeddings)
        x_metadata = data['metadata']
        x = torch.cat((x_images, x_embeddings, x_metadata), dim = 1)
        x = self.classifier(x)
        return x

        
class CNN(nn.Module):
  """
  Convolutional Neural Network (CNN) with an 
  - Convolutional layer for three input channels with 8 kernels (3x3) and padding 1.
  Use batch normalization before apply a ReLU activation function. Max pooling and 
  dropout are applied
  - Global max and average pooling is applied
  - 8 neuron fully-connected layer for the 16 output features from above steps
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

    
def ResNet18_layer4_fc16():
    """
    Initializes a pretrained ResNet18 model for transfer learning with the following modifications:
    1. Loads weights pretrained on ImageNet
    2. Freezes all layers except layer4 
    3. Replaces the final fully-connected layer for binary classification
    """
    # build a model based on ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')
    #model.eval()  # Set to evaluation mode

    # Freeze early layers, keep later layers trainable
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:  # Freeze everything except layer4 and fc
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Change classificator by smaller one
    num_features = model.fc.in_features # input feature to classificator
    model.fc = torch.nn.Linear(num_features, 16) # Two levels engagement

    return model

def ResNet18_layer4_nofc():
    """
    Initializes a pretrained ResNet18 model for transfer learning with the following modifications:
    1. Loads weights pretrained on ImageNet
    2. Freezes all layers except layer4
    3. Remove the final fully-connected layer
    """
    # build a model based on ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')
    #model.eval()  # Set to evaluation mode

    # Freeze early layers, keep later layers trainable
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:  # Freeze everything except layer4 and fc
            param.requires_grad = False
        else:
            param.requires_grad = True

    model.fc = torch.nn.Identity()

    return model
    
class dual_ResNet18_layer4_fc16(nn.Module):
    def __init__(self, dropout_rate):
        super(dual_ResNet18_layer4_fc16, self).__init__()
        
        # Images branch
        self.model_images = ResNet18_layer4_fc16()
        
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
          nn.Linear(16, 2)
        )
    
    def forward(self, data):
        x_images = data['img']
        #x_images = self.model_images(x_images).squeeze()
        x_images = self.model_images(x_images)
        x_fcnn = data['meta']
        x_fcnn = self.FCNN_fcLayer1(x_fcnn)
        x = torch.cat((x_images, x_fcnn), dim = 1)
        x = self.class_fcLayer1(x)
        return x
    
class dual_ResNet18_layer4(nn.Module):
    def __init__(self, dropout_rate):
        super(dual_ResNet18_layer4, self).__init__()
        
        # Images branch
        self.model_images = ResNet18_layer4_nofc()
        
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
        x_images = data['img']
        #x_images = self.model_images(x_images).squeeze()
        x_images = self.model_images(x_images)
        x_fcnn = data['meta']
        x_fcnn = self.FCNN_fcLayer1(x_fcnn)
        x = torch.cat((x_images, x_fcnn), dim = 1)
        x = self.class_fcLayer1(x)
        return x

      
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