import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score
from scipy.stats import uniform
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self, layers):
        print(layers)
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, int(layers))
        self.fc2 = nn.Linear(int(layers), 10)
        

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
testloader = []

def train(model, num_epoch, trainloader=None, valloader=None, optimizer=None, early_stopping_patience=None):
    if trainloader is None:
        trainloader, _, _ = load_data()
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in tqdm(range(num_epoch)):
        model.train()
        for images, labels in tqdm(trainloader, leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Validation phase
        if valloader and early_stopping_patience:
            val_acc = test(model, valloader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

def test(model, testloader=None):
    if testloader is None:
        _, testloader = load_data()
        
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(testloader, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def load_data(batch_size=64, val_split=None):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                          download=True, transform=transform)
    
    if val_split:
        val_size = int(len(trainset) * val_split)
        train_size = len(trainset) - val_size
        trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    else:
        valloader = None

    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                           batch_size=batch_size, 
                                           shuffle=False)

    return trainloader, valloader, testloader

if __name__ == "__main__":
    train(5)