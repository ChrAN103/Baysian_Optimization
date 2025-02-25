import GPyOpt
import numpy as np
from model import CNN, load_data, train, test
import torch
import torch.nn as nn
import torch.optim as optim

def objective_function(x):
    # Extract parameters
    layers = float(x[:, 0])
    
    # Initialize model
    model = CNN(layers)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Load data with current batch size
    trainloader, testloader = load_data(batch_size=64)
    
    # Train model
    train(model, 5, trainloader=trainloader, optimizer=optimizer)
    
    # Test model
    accuracy = test(model, testloader=testloader)
    
    # GPyOpt minimizes, so return negative accuracy
    return -accuracy

# Define the parameter space
bounds = [
    {'name': 'layers', 'type': 'discrete', 'domain': (64,128,256,512)}
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