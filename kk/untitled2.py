class Train:
        def __init__(self):
        """
        Initialize Train object with empty lists for losses and accuracies.
        """
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def update(self, train_loss, train_acc):
        """
        Add new training loss and accuracy values to the tracking lists.
        """
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
    
    def to_dict(self):
        """
        Return training data as a dictionary.        
        """
        return {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs
        }