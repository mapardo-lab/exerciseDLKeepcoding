import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import random
import numpy  as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def set_random_seed(seed=42):
  """
  Fixed seeds for reproducibility
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

def train_model(model, criterion, optimizer, num_epochs, trainloader, valloader, 
                device, testloader=None, l1_lambda = None, scheduler = None, verbose = True):
  """
  Train model and plot metrics from training

  Args:
    model: Neural network model to train
    criterion: Objective loss function
    optimizer: Parameter optimizer
    num_epochs: Number of epochs for training
    trainloader: Training data loader
    valloader: Validation data loader
    device: Training device ('cuda' or 'cpu')
    testloader: Test data loader
    l1_lambda: Parameter L1 for regularization
    scheduler: LR scheduler instance
  """
  result = None
  model.to(device)

  train_losses, train_accs, val_losses, val_accs = [], [], [], []
  for epoch in range(num_epochs):
      loss, acc , lr = train_epoch(model, device, trainloader, criterion, optimizer, l1_lambda=l1_lambda, scheduler=scheduler)
      val_loss, val_acc = eval_epoch(model, device, valloader, criterion)
      train_losses.append(loss)
      train_accs.append(acc)
      val_losses.append(val_loss)
      val_accs.append(val_acc)
      if verbose:
        print(f'Epoch {epoch+1}, Loss: {loss}, Acc: {acc}, Val Loss: {val_loss}, Val Acc: {val_acc}, LR: {lr}')
  if testloader is not None:
    accuracy = evaluate_model(model, testloader, device)
    accuracy100 = accuracy*100
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs, test_acc = accuracy100) 
  elif verbose:
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs) 
  else:
    result = {'train_losses':train_losses, 
              'train_accs': train_accs, 
              'val_losses': val_losses, 
              'val_accs': val_accs}

  return result

def train_epoch(model: nn.Module, device: torch.device, train_loader: DataLoader, 
                criterion, optimizer, l1_lambda=None, scheduler=None):
  """
  Train neural network for one epoch and return the training metrics.
    
  Args:
    model: Neural network model to train
    device: Training device ('cuda' or 'cpu')
    train_loader: Training data loader
    criterion: Objective loss function
    optimizer: Parameter optimizer
    l1_lambda: Parameter L1 for regularization
    scheduler: LR scheduler instance
        
  Returns:
    train_loss: Average loss on the training set  
    train_acc: Training set accuracy (%)  
    current_lr: Current learning rate after scheduling
  """
  model.train()
  train_loss = 0
  correct = 0
  total = 0
    
  for batch_idx, data in enumerate(train_loader):
    for key, value in data.items():
      data[key] = value.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data['target'])
    if l1_lambda is not None:
      l1_norm = sum(p.abs().sum() for p in model.parameters())
      loss += l1_lambda * l1_norm
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    _, predicted = output.max(1)
    total += data['target'].size(0)
    correct += predicted.eq(data['target']).sum().item()

  train_loss /= len(train_loader)
  train_acc = 100. * correct / total

  if scheduler is not None:
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    return train_loss, train_acc, current_lr
  else:
    return train_loss, train_acc, optimizer.param_groups[0]['lr']


def eval_epoch(model: nn.Module, device: torch.device, 
               val_loader: DataLoader, criterion):
  """
  Evaluates network model on validation data and returns metrics
    
  Args:
    model: Neural network model to evaluate
    device: Hardware device for evaluation (CPU/GPU)
    val_loader: Validation DataLoader
    criterion: Loss function to use
        
  Returns:
    val_loss (float): Average loss on validation batches
    val_acc (float): Validation accuracy percentage
  """
  model.eval()
  val_loss = 0
  correct = 0
  total = 0
    
  with torch.no_grad():
    for data in val_loader:
      for key, value in data.items():
        data[key] = value.to(device)
      output = model(data)
      loss = criterion(output, data['target'])
      val_loss += loss.item()
      _, predicted = output.max(1)
      total += data['target'].size(0)
      correct += predicted.eq(data['target']).sum().item()

  val_loss /= len(val_loader)
  val_acc = 100. * correct / total

  return val_loss, val_acc

def evaluate_model(model, testloader, device):
  """
  Evaluates network model on a test dataset.

  Args:
    model: Trained neural network model to evaluate.
    testloader: DataLoader containing the test dataset.
    device: Device to perform evaluation on ('cuda' or 'cpu').

  Returns:
    float: Classification accuracy percentage on the test set.
  """
  model.eval()
  total_predicted = np.array([])
  total_labels = np.array([])
  with torch.no_grad():
    for data in testloader:
      for key, value in data.items():
        data[key] = value.to(device)
      outputs = model(data)
      _, predicted = torch.max(outputs.data, 1)
      total_predicted = np.concatenate([total_predicted,predicted.cpu().numpy()])
      total_labels = np.concatenate([total_labels,data['target'].cpu().numpy()])
    
  accuracy = accuracy_score(total_labels, total_predicted)
  print('Confussion matrix:')
  print(f'{confusion_matrix(total_labels, total_predicted)}')
  print('\nClassification report')
  print(f'{classification_report(total_labels, total_predicted)}')
  print(f'Accuracy score: {accuracy:.2f}')

  return accuracy

def plot_training_curves(train_losses, val_losses, train_accs, 
                         val_accs, num_epochs, test_acc=None):
  """
  From the model training output, the training progress is plotted 
  for loss function and accuracy values. Optionally, accuracy for
  the test is also plotted.
  """
  plt.style.use("ggplot")
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(range(num_epochs), train_losses, label="Train Loss")
  plt.plot(range(num_epochs), val_losses, label="Validation Loss")
  plt.title("Training and Validation Loss")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(range(num_epochs), train_accs, label="Train Accuracy")
  plt.plot(range(num_epochs), val_accs, label="Validation Accuracy")
  if test_acc is not None:
    plt.axhline(y=test_acc, color='red', linestyle='--', label='Test Accuracy')
  plt.title("Training and Validation Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.tight_layout()
  plt.show()