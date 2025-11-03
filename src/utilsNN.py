import torch
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
    def get_cnn_simple_classifier(input_channels=3, output_size=2):
        """Simple convolutional Neural Network (CNN)"""
        return nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        
            # Classifier
            nn.AdaptativeAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, output_size)
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
    
    @staticmethod
    def last_layer_input(model):
        """Return the input feature size of the last layer in a PyTorch model."""
        last_layer = list(model.children())[-1]
        return last_layer.in_features
        
    
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
        
class ResNet18_pretrain(nn.Module):
    """
    Pre-trained ResNet-18 backbone for image classification
    """
    def __init__(self, num_classes):
        super(ResNet18_pretrain, self).__init__()
        self.network = CommonBlocks.get_ResNet18_pre_classifier(num_classes)

    def forward(self, data):
        x = data['images'] 
        x = self.network(x)
        return x 

class ResNet18_layer4(nn.Module):
    """
    Fine-tunning ResNet-18 backbone for image classification
    """
    def __init__(self, num_classes):
        super(ResNet18_layer4, self).__init__()
        self.network = CommonBlocks.get_ResNet18_layer4_classifier(num_classes)

    def forward(self, data):
        x = data['images'] 
        x = self.network(x)
        return x 

class MultiModal3_pre_Class2(nn.Module):
    """A multi-modal neural network for binary classification using three data modalities.

    This model processes three different types of input data through separate branches:
    images through a ResNet18 backbone, embeddings through a fully connected network,
    and metadata through direct concatenation. The features from all three branches
    are combined for final classification.
    """
    def __init__(self, metadata_input, embeddings_input, dropout_rate):
        super(MultiModal3_pre_Class2, self).__init__()
        
        # Images branch
        self.branch_images = CommonBlocks.get_ResNet18_pre_classifier()
        images_size = CommonBlocks.last_layer_input(self.branch_images)        
        self.branch_images.fc = nn.Identity()
        
        # Embeddings branch
        fcnn_pca_classifier = CommonBlocks.get_fcnn_pca_classifier(embeddings_input, dropout=dropout_rate)
        embeddings_size = CommonBlocks.last_layer_input(fcnn_pca_classifier)        
        self.branch_embeddings = nn.Sequential(*list(fcnn_pca_classifier.children())[:-1])
        
        total_features = images_size + embeddings_size + metadata_input
    
        # Classificator: Fully connected layer
        self.classifier = CommonBlocks.get_fcnn_pca_classifier(total_features, 2, dropout_rate)
    
    def forward(self, data):
        x_images = data['images']
        x_images = self.branch_images(x_images)
        x_embeddings = data['embeddings']
        x_embeddings = self.branch_embeddings(x_embeddings)
        x_metadata = data['metadata']
        x = torch.cat((x_images, x_embeddings, x_metadata), dim = 1)
        x = self.classifier(x)
        return x

class MultiModal3_layer4_Class2(nn.Module):
    """A multi-modal neural network for binary classification using three data modalities.

    This model processes three different types of input data through separate branches:
    images through a ResNet18 backbone with layer4 optimization, embeddings through a 
    fully connected network, and metadata through direct concatenation. 
    The features from all three branches are combined for final classification.
    """
    def __init__(self, metadata_input, embeddings_input, dropout_rate):
        super(MultiModal3_layer4_Class2, self).__init__()
        
        # Images branch
        self.branch_images = CommonBlocks.get_ResNet18_layer4_classifier()
        images_size = CommonBlocks.last_layer_input(self.branch_images)        
        self.branch_images.fc = nn.Identity()
        
        # Embeddings branch
        fcnn_pca_classifier = CommonBlocks.get_fcnn_pca_classifier(embeddings_input, dropout=dropout_rate)
        embeddings_size = CommonBlocks.last_layer_input(fcnn_pca_classifier)        
        self.branch_embeddings = nn.Sequential(*list(fcnn_pca_classifier.children())[:-1])
        
        total_features = images_size + embeddings_size + metadata_input
    
        # Classificator: Fully connected layer
        self.classifier = CommonBlocks.get_fcnn_pca_classifier(total_features, 2, dropout_rate)
    
    def forward(self, data):
        x_images = data['images']
        x_images = self.branch_images(x_images)
        x_embeddings = data['embeddings']
        x_embeddings = self.branch_embeddings(x_embeddings)
        x_metadata = data['metadata']
        x = torch.cat((x_images, x_embeddings, x_metadata), dim = 1)
        x = self.classifier(x)
        return x

