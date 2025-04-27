import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

print("Loading data")

# Load the FashionMNIST data
def get_fashionmnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    return dataset

# Function to perform subsampling 50% from each class
def subsample_10_percent_per_class(dataset):
    """
    Subsample 10% of the data from each class.
    dataset: The full dataset 
    Returns: A list of indices for the subsampled dataset
    """

    #Get all labels in dataset
    all_labels = np.array([dataset[i][1] for i in range(dataset.data.shape[0])])

    sampled_indices = []
    #Iterate through classes/labels
    for j in range(10):
        #Find indices where label is a certain value
        idx_array = np.where(all_labels == j)[0]
        #Keep 50 percent of these samples, add to sampled_indices list
        random_idxs_class = np.random.randint(0, len(idx_array), size = int(np.round(len(idx_array)/10)))
        sampled_indices.extend(idx_array[random_idxs_class])

    return sampled_indices

# Define the Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# Noam Scheduler (from "Attention is All You Need")
class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** -0.5) * min(step ** -0.5, step * self.warmup ** -1.5)
    
class CosineAnnealing:
    def __init__(self, max_lr, warmup, total_steps, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.max_lr = max_lr
        self.total_steps = total_steps
        self._rate = 0
    
    def step(self):
        self._step += 1
        rate = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        if step <= self.warmup:
            return 1e-4 + (self.max_lr - 1e-4) * (step / self.warmup)
        else:
            return self.max_lr * np.cos((math.pi / 2) * ((step - self.warmup) / (self.total_steps - self.warmup))) + 1e-6

# Train and Validate the Model
def train_and_validate(train_loader, val_loader, model, optimizer, scheduler_type, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_acc, val_acc = [], []
    train_loss, val_loss = [], []
    global_step = 0

    if scheduler_type == 'noam':
        scheduler = NoamOpt(model_size=512, factor=56.29, warmup=(len(train_loader)*epochs) // 5, optimizer=optimizer)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealing(max_lr = 0.0394, warmup=(len(train_loader)*epochs) // 5, total_steps=len(train_loader)*epochs, optimizer=optimizer)
    else:
        raise ValueError("Unsupported scheduler type.")

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            global_step += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            scheduler.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss_epoch, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_epoch += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc.append(100 * correct / total)
        val_loss.append(val_loss_epoch / len(val_loader))

    return train_acc, train_loss, val_acc, val_loss

# Cross-validation for hyperparameter tuning
def cross_validate_model(dataset, model_fn, params, k_folds=5, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = {}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Training fold {fold+1}/{k_folds}...")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=params['batch_size'], shuffle=False)

        model = model_fn(dropout=params['dropout']).to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=params['momentum'], weight_decay=params['weight_decay'])

        if params['lr_scheduler'] == 'noam':
            train_acc, train_loss, val_acc, val_loss = train_and_validate(train_loader, val_loader, model, optimizer, 'noam', epochs)
        else:
            train_acc, train_loss, val_acc, val_loss = train_and_validate(train_loader, val_loader, model, optimizer, 'cosine', epochs)

        results[fold] = {
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_loss': val_loss
        }

    return results

# Grid Search with Batch Size and Other Parameters
def grid_search(dataset, model_fn, param_grid, k_folds=5, epochs=5):
    best_params = None
    best_val_acc = 0

    all_results = {}

    # Iterate through each hyperparameter combination
    for dropout in param_grid['dropout']:
        for momentum in param_grid['momentum']:
            for batch_size in param_grid['batch_size']:
                for weight_decay in param_grid['weight_decay']:
                    for lr_scheduler in param_grid['lr_scheduler']:
                        params = {
                            'dropout': dropout,
                            'momentum': momentum,
                            'batch_size': batch_size,
                            'weight_decay': weight_decay,
                            'lr_scheduler': lr_scheduler
                        }

                        print(f"Evaluating: {params}")

                        results = cross_validate_model(Subset(dataset, subsample_10_percent_per_class(dataset)), model_fn, params, k_folds, epochs)

                        avg_val_acc = np.mean([results[fold]['val_acc'][-1] for fold in range(k_folds)])

                        print(f"Avg Validation Accuracy: {avg_val_acc:.2f}%")

                        if avg_val_acc > best_val_acc:
                            best_val_acc = avg_val_acc
                            best_params = params

                        # Store results for all hyperparameters for plotting later
                        hyperparams_key = f"Dropout: {dropout}, Momentum: {momentum}, Batch Size: {batch_size}, WD: {weight_decay}, LR Scheduler: {lr_scheduler}"
                        all_results[hyperparams_key] = {
                            'val_acc': [np.mean([results[fold]['val_acc'][-1] for fold in range(k_folds)])],
                            'train_acc': [np.mean([results[fold]['train_acc'][-1] for fold in range(k_folds)])],
                            'train_loss': [np.mean([results[fold]['train_loss'][-1] for fold in range(k_folds)])],
                            'val_loss': [np.mean([results[fold]['val_loss'][-1] for fold in range(k_folds)])]
                        }

    print(f"Best Params: {best_params}")
    return best_params, all_results

# Run the Experiment
def run_experiment():
    dataset = get_fashionmnist_data()

    param_grid = {
        'dropout': [0.3, 0.5, 0.7],
        'momentum': [0.8, 0.9, 0.95],
        'batch_size': [32, 64, 128],
        'weight_decay': [0.0, 0.0005, 0.001],
        'lr_scheduler': ['noam', 'cosine']
    }

    print("starting grid search")
    best_params, all_results = grid_search(dataset, SimpleCNN, param_grid, k_folds=5, epochs=5)

    print(f"Best Hyperparameters: {best_params}")

    # Plot validation accuracy for each hyperparameter combination
    plt.figure(figsize=(10, 6))
    for key, result in all_results.items():
        plt.plot(result['val_acc'], label=key)
    plt.xlabel('Hyperparameter Combinations')
    plt.ylabel('Validation Accuracy')
    plt.legend()

    plt.savefig("plots/val_accuracy_plot.png") #if you want to save the figure, e.g. running remotely and you'll look later
    plt.show()                                  # if you only want to save it, and not have it display, you can comment this line out

    return best_params, all_results

if __name__ == '__main__':
    best_params, all_results = run_experiment()
