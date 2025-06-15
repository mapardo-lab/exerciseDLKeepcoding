import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import random
import requests
from PIL import Image
from io import BytesIO
import numpy  as np  
import pandas as pd
import seaborn as sns
#from adjustText import adjust_text
from scipy.stats import ttest_ind

#############################
# Funciones de visualización#
#############################
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs, test_acc=None):
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


#############################
# Funciones de entrenamiento#
#############################

# Definimos la función para entrenar una época (sacada del notebook II)
def train_epoch(model: nn.Module, device: torch.device, train_loader: DataLoader, 
                criterion, optimizer, l1_lambda=None, scheduler=None):
    """
    Entrena una época de la red neuronal y devuelve las métricas de entrenamiento.
    
    Args:
        model: Modelo de red neuronal a entrenar
        device: Dispositivo donde se realizará el entrenamiento (CPU/GPU)
        train_loader: DataLoader con los datos de entrenamiento
        criterion: Función de pérdida a utilizar
        optimizer: Optimizador para actualizar los pesos
        scheduler: Scheduler para ajustar el learning rate
        
    Returns:
        train_loss: Pérdida promedio en el conjunto de entrenamiento
        train_acc: Precisión en el conjunto de entrenamiento (%)
        current_lr: Learning rate actual después del scheduler
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if l1_lambda is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100. * correct / total

    # Aplicar el scheduler después de cada época
    if scheduler is not None:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        return train_loss, train_acc, current_lr
    else:
        return train_loss, train_acc, optimizer.param_groups[0]['lr']


def eval_epoch(model: nn.Module, device: torch.device, val_loader: DataLoader, 
               criterion):
    """
    Evalúa el modelo en el conjunto de validación.
    
    Args:
        model: Modelo de red neuronal a evaluar
        device: Dispositivo donde se realizará la evaluación (CPU/GPU)
        val_loader: DataLoader con los datos de validación
        criterion: Función de pérdida a utilizar
        
    Returns:
        val_loss: Pérdida promedio en el conjunto de validación
        val_acc: Precisión en el conjunto de validación (%)
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc

def evaluate_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputdata, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputdata)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100. * correct / total

def train_model(model, criterion, optimizer, num_epochs,
            trainloader, valloader, device, testloader=None, 
            l1_lambda = None, scheduler = None):
  model.to(device)

  train_losses, train_accs, val_losses, val_accs = [], [], [], []
  for epoch in range(num_epochs):
      loss, acc , lr = train_epoch(model, device, trainloader, criterion, optimizer, l1_lambda=l1_lambda, scheduler=scheduler)
      val_loss, val_acc = eval_epoch(model, device, valloader, criterion)
      train_losses.append(loss)
      train_accs.append(acc)
      val_losses.append(val_loss)
      val_accs.append(val_acc)
      print(f'Epoch {epoch+1}, Loss: {loss}, Acc: {acc}, Val Loss: {val_loss}, Val Acc: {val_acc}, LR: {lr}')
  if testloader is not None:
    accuracy = evaluate_model(model, testloader, device)
    print(f"\n🎯 Precisión en Test: {accuracy:.2f}%")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs, test_acc = accuracy) 
  else:
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs) 

def parameters_network(model):
  # Contamos el número total de parámetros del modelo
  total_params = sum(p.numel() for p in model.parameters())
  print(f'Número total de parámetros: {total_params:,}')

  # Desglose de parámetros por capa
  print('\nDesglose de parámetros por capa:')
  for name, param in model.named_parameters():
    print(f'{name}: {param.numel():,} parámetros')

def plot_density(df: pd.DataFrame, features: list , n_rows: int, n_cols: int):
    """
    From dataframe plot several density plots for the features contained in the list.
    Plots are distributed in rows and columns
    """
    plt.figure(figsize=(8 * n_cols, 4 * n_rows))  # Dynamic figure size
    for i, feature in enumerate(features,start=1):
        plt.subplot(n_rows, n_cols, i)
        sns.kdeplot(df[feature], fill=True, color='skyblue', alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel('Density')
    plt.show()

def plot_bars(df: pd.DataFrame, features: list , n_rows: int, n_cols: int):
    """
    From dataframe plot several bar plots for the features contained in the list.
    Plots are distributed in rows and columns
    """
    plt.figure(figsize=(6 * n_cols, 4 * n_rows))  # Dynamic figure size
    for i, feature in enumerate(features, start=1):
        counts = df[feature].value_counts()
        counts = counts.loc[sorted(counts.index)]
        plt.subplot(n_rows, n_cols, i)
        # Plot value counts as bars
        counts.plot(kind='bar', color='skyblue', edgecolor='black')
        # Customize
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()  # Prevent label clipping
    plt.show()

# Fijamos todas las semillas aleatorias para reproducibilidad
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
