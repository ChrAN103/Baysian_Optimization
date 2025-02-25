import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import pandas as pd
from datetime import datetime

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 256
epochs = 3
learning_rate = 0.001
subset_size = 5000

# Data transformation and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create a subset of the training data
np.random.seed(42)
train_indices = np.random.choice(len(train_full_dataset), subset_size, replace=False)
train_dataset = Subset(train_full_dataset, train_indices)

print(f"Training on {len(train_dataset)} samples (reduced from {len(train_full_dataset)})")

# Simple data loading
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model with a single dropout layer
class Net(nn.Module):
    def __init__(self, conv1_channels=16, conv2_channels=32, dropout_rate=0.3, hidden_size=64):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, 3, 1, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout_rate)  # Single dropout layer
        self.fc1 = nn.Linear(conv2_channels * 14 * 14, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)  # Single dropout after the first fully connected layer
        x = self.fc2(x)
        return x

# Training function
def train(model, device, train_loader, optimizer, epoch, verbose=True):
    model.train()
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if verbose and batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx * len(data) / len(train_loader.dataset):.0f}%)]\tLoss: {loss.item():.4f}')
    
    if verbose:
        print(f"Epoch {epoch} training time: {time.time() - start_time:.2f} seconds")

# Testing function
def test(model, device, test_loader, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    if verbose:
        print(f'Test set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return accuracy

# Storage for optimization results
optimization_results = []

# Function for Bayesian optimization
def train_and_evaluate(learning_rate=0.001, batch_size=256, dropout_rate=0.3, 
                       conv1_channels=16, conv2_channels=32, hidden_size=64):
    """
    Function to be optimized by Bayesian optimization.
    Returns the test accuracy for the given hyperparameters.
    """
    # Convert numerical parameters to appropriate types
    batch_size = int(batch_size)
    conv1_channels = int(conv1_channels)
    conv2_channels = int(conv2_channels)
    hidden_size = int(hidden_size)
    
    # Store hyperparameters for logging
    params = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'conv1_channels': conv1_channels,
        'conv2_channels': conv2_channels,
        'hidden_size': hidden_size,
    }
    
    print(f"\nTesting hyperparameters: {params}")
    
    # Create model with the given hyperparameters
    model = Net(
        conv1_channels=conv1_channels, 
        conv2_channels=conv2_channels, 
        dropout_rate=dropout_rate, 
        hidden_size=hidden_size
    ).to(device)
    
    # Initialize optimizer with the given learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loaders with the given batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Train for a fixed number of epochs
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, verbose=False)
    
    # Get final test accuracy
    accuracy = test(model, device, test_loader, verbose=True)
    
    # Store results
    result = params.copy()
    result['accuracy'] = accuracy
    result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result['iteration'] = len(optimization_results) + 1
    optimization_results.append(result)
    
    return accuracy

# Plot function for optimization results
def plot_optimization_results(results_df):
    # Create plots directory if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Plot accuracy over iterations
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['iteration'], results_df['accuracy'], 'bo-', linewidth=2, markersize=8)
    
    # Add moving average line
    window_size = min(5, len(results_df))
    if window_size > 1:
        rolling_mean = results_df['accuracy'].rolling(window=window_size).mean()
        plt.plot(results_df['iteration'][window_size-1:], rolling_mean[window_size-1:], 
                'r--', linewidth=2, label=f'{window_size}-point Moving Average')
    
    # Add best accuracy line
    plt.axhline(y=results_df['accuracy'].max(), color='g', linestyle='--', 
                label=f'Max Accuracy: {results_df["accuracy"].max():.2f}%')
    
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Improvement with Bayesian Optimization')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/accuracy_vs_iteration.png')
    plt.close()
    
    # Plot parameter importance
    plt.figure(figsize=(14, 10))
    param_cols = ['learning_rate', 'batch_size', 'dropout_rate', 
                  'conv1_channels', 'conv2_channels', 'hidden_size']
    
    for i, param in enumerate(param_cols):
        plt.subplot(2, 3, i+1)
        
        # Calculate correlation
        correlation = results_df[param].corr(results_df['accuracy'])
        
        # Create scatter plot
        plt.scatter(results_df[param], results_df['accuracy'])
        
        # Add trend line
        z = np.polyfit(results_df[param], results_df['accuracy'], 1)
        p = np.poly1d(z)
        plt.plot(sorted(results_df[param]), p(sorted(results_df[param])), "r--", alpha=0.8)
        
        plt.xlabel(param)
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy vs {param} (r={correlation:.2f})')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/parameter_influence.png')
    plt.close()
    
    # Create a heatmap of parameter correlations
    plt.figure(figsize=(10, 8))
    corr_data = results_df[param_cols + ['accuracy']].corr()
    plt.imshow(corr_data, cmap='coolwarm', interpolation='none', aspect='auto')
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(corr_data.columns)), corr_data.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_data.columns)), corr_data.columns)
    
    # Add correlation values
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            text = plt.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black")
    
    plt.title('Parameter Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/parameter_correlation.png')
    plt.close()
    
    # Return the best parameters
    best_idx = results_df['accuracy'].idxmax()
    return results_df.iloc[best_idx].to_dict()

# Run a grid search over one parameter to compare with Bayesian optimization
def run_grid_search(param_name, param_values, fixed_params=None):
    """Run a grid search over a specific parameter."""
    fixed_params = fixed_params or {}
    grid_results = []
    
    print(f"\n{'='*50}")
    print(f"Running grid search for parameter: {param_name}")
    print(f"{'='*50}")
    
    for value in param_values:
        # Set up parameters
        params = fixed_params.copy()
        params[param_name] = value
        
        print(f"\nTesting {param_name}={value}")
        
        # Convert parameters to appropriate types
        if param_name in ['batch_size', 'conv1_channels', 'conv2_channels', 'hidden_size']:
            params[param_name] = int(params[param_name])
        
        # Create and train model
        model = Net(
            conv1_channels=params.get('conv1_channels', 16), 
            conv2_channels=params.get('conv2_channels', 32), 
            dropout_rate=params.get('dropout_rate', 0.3), 
            hidden_size=params.get('hidden_size', 64)
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
        
        train_loader = DataLoader(train_dataset, 
                                  batch_size=params.get('batch_size', 256), 
                                  shuffle=True)
        test_loader = DataLoader(test_dataset, 
                                batch_size=params.get('batch_size', 256), 
                                shuffle=False)
        
        # Train for fixed epochs
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch, verbose=False)
        
        # Test and record accuracy
        accuracy = test(model, device, test_loader, verbose=True)
        
        result = params.copy()
        result['accuracy'] = accuracy
        grid_results.append(result)
    
    # Create a DataFrame from results
    grid_df = pd.DataFrame(grid_results)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(grid_df[param_name], grid_df['accuracy'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel(param_name)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Grid Search Results: Accuracy vs {param_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/grid_search_{param_name}.png')
    plt.close()
    
    # Save results
    grid_df.to_csv(f'plots/grid_search_{param_name}.csv', index=False)
    
    return grid_df

if __name__ == "__main__":
    # Train baseline model first
    print("Training baseline model...")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    baseline_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
    
    baseline_accuracy = test(model, device, test_loader)
    baseline_time = time.time() - baseline_start_time
    
    print(f"Baseline training completed in {baseline_time:.2f} seconds.")
    print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    
    # Save baseline model
    torch.save(model.state_dict(), "mnist_cnn_baseline.pt")
    
    # Run a grid search for learning rate to compare with Bayesian optimization
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    grid_results = run_grid_search('learning_rate', learning_rates)
    best_grid_lr = grid_results.loc[grid_results['accuracy'].idxmax()]['learning_rate']
    best_grid_acc = grid_results['accuracy'].max()
    
    print(f"\nBest learning rate from grid search: {best_grid_lr}")
    print(f"Best accuracy from grid search: {best_grid_acc:.2f}%")
    
    # Define parameter bounds for Bayesian optimization
    pbounds = {
        'learning_rate': (0.0001, 0.01),
        'batch_size': (32, 512),
        'dropout_rate': (0.1, 0.6),
        'conv1_channels': (8, 32),
        'conv2_channels': (16, 64),
        'hidden_size': (32, 128)
    }
    
    # Run Bayesian Optimization
    print(f"\n{'='*50}")
    print("Starting Bayesian Optimization...")
    print(f"{'='*50}")
    
    # Initialize optimizer
    optimizer = BayesianOptimization(
        f=train_and_evaluate,
        pbounds=pbounds,
        random_state=42
    )
    
    # Run optimization
    optimizer.maximize(init_points=5, n_iter=15)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(optimization_results)
    
    # Plot results
    print("\nPlotting results...")
    best_params = plot_optimization_results(results_df)
    
    # Save all results
    results_df.to_csv('plots/bayesian_optimization_results.csv', index=False)
    
    print("\nBest parameters from Bayesian optimization:")
    for param, value in best_params.items():
        if param not in ['accuracy', 'timestamp', 'iteration']:
            print(f"{param}: {value}")
    
    print(f"\nBest accuracy from Bayesian optimization: {best_params['accuracy']:.2f}%")
    print(f"Improvement over baseline: {best_params['accuracy'] - baseline_accuracy:.2f}%")
    print(f"Improvement over grid search: {best_params['accuracy'] - best_grid_acc:.2f}%")
    
    print("\nResults saved to 'plots' directory.")