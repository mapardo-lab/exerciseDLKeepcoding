import numpy as np
import torch
from utils import info_object

#class ModelTrainL1(ModelTrain):
#    def __init__(self, model, device, criterion, optimizer, scoring, params, fixed_params,):
#        super().__init__(model, device, criterion, optimizer, scoring, params, fixed_params)

class ModelTrain():
    def __init__(self, train_config, params, fixed_params = None):
        self.architecture = train_config['model']
        self.arch_params = params['model'] | fixed_params['model']
        self.model = self.architecture(**self.arch_params)
        self.criterion = train_config['criterion'](**params['criterion'], **fixed_params['criterion'])
        self.optimizer = train_config['optimizer'](params = self.model.parameters(), **params['optimizer'], **fixed_params['optimizer'])
        self.device = train_config['device']
        self.model.to(self.device)
        self.scoring = train_config['scoring']
        self.results_train = ResultTrain(self.scoring)
        self.results_val = ResultTrain(self.scoring)

    def train_epoch(self, train_loader): 
        """
        Train neural network for one epoch and return the training metrics.
        """
        self.model.train()
        train_loss = 0
        y_pred = np.array([])
        y_true = np.array([])

        for data in train_loader:
            for key, value in data.items():
                data[key] = value.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data['target'])
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            y_true = np.concatenate([y_true, data['target'].cpu().numpy()])
            y_pred = np.concatenate([y_pred, predicted.cpu().numpy()])

        self.results_train.update(train_loss, y_true, y_pred)

    def eval_model(self, val_loader): 
        """
        Evaluates network model on validation data and returns metrics
        """
        self.model.eval()
        val_loss = 0
        y_pred = np.array([])
        y_true = np.array([])

        with torch.no_grad():
            for data in val_loader:
                for key, value in data.items():
                    data[key] = value.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, data['target'])
                val_loss += loss.item()
                _, predicted = output.max(1)
                y_true = np.concatenate([y_true, data['target'].cpu().numpy()])
                y_pred = np.concatenate([y_pred, predicted.cpu().numpy()])
        self.results_val.update(val_loss, y_true, y_pred)

    def train_model(self, num_epochs, train_loader):
        """
        Train model and plot metrics from training
        """
        self.model.to(self.device)

        for epoch in range(num_epochs):
            self.train_epoch(train_loader)

        #if testloader is not None:
        #    accuracy = evaluate_model(model, testloader, device)
        #    plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs, test_acc = accuracy) 

        #return result

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename + ".pth")

class ResultTrain:
    """
    Tracks and accumulates training and validation metrics throughout the training process.
    """
    def __init__(self, scoring):
        """
        Initialize Train object with empty lists for losses and accuracies.
        """
        self.scoring = scoring
        self.results = {}
        for score_name, _ in scoring.items():
            self.results['loss'] = []
            self.results[score_name] = []

    def update(self, loss, y_true, y_pred):
        new_results = self.get_scores(loss, y_true, y_pred)
        for score_name, score in new_results.items():
            self.results[score_name].append(score)

    def get_scores(self, loss, y_true, y_pred):
        """
        Computes and returns a dictionary of evaluation metrics including loss and specified scoring functions.
        """
        results = {'loss': loss}
        for score_name, score_func in self.scoring.items():
            results[score_name] = score_func(y_true, y_pred)
        return results
    
def plot_training_curves(train_results, val_results):
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
    