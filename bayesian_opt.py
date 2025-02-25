import GPyOpt
import numpy as np
from model import CNN, load_data, train, test
import torch
import torch.nn as nn
import torch.optim as optim

def objective_function(x):
    # Extract parameters
    learning_rate = float(x[:, 0])
    batch_size = int(x[:, 1])
    epochs = int(x[:, 2])
    
    # Initialize model
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load data with current batch size
    trainloader, testloader = load_data(batch_size=batch_size)
    
    # Train model
    train(model, epochs, trainloader=trainloader, optimizer=optimizer)
    
    # Test model
    accuracy = test(model)
    
    # GPyOpt minimizes, so return negative accuracy
    return -accuracy

# Define the parameter space
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64, 128)},
    {'name': 'epochs', 'type': 'discrete', 'domain': (3, 5, 7, 10)}
]

def main():
    # Initialize Bayesian Optimization
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective_function,
        domain=bounds,
        model_type='GP',
        acquisition_type='EI',
        maximize=False,
        initial_design_numdata=5
    )
    
    # Run optimization
    optimizer.run_optimization(max_iter=15)
    
    # Print results
    print("Optimal parameters:")
    print(f"Learning rate: {optimizer.x_opt[0]:.6f}")
    print(f"Batch size: {int(optimizer.x_opt[1])}")
    print(f"Epochs: {int(optimizer.x_opt[2])}")
    print(f"Best accuracy: {-optimizer.fx_opt:.4f}")
    
    # Plot convergence
    optimizer.plot_convergence()

if __name__ == "__main__":
    main() 