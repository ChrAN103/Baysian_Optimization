import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import GPyOpt
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


class CNN(nn.Module):
    def __init__(self, channels, fc_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear((channels * 2) * 8 * 8, fc_size)
        self.fc2 = nn.Linear(fc_size, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_and_evaluate(channels, fc_size):
    model = CNN(int(channels), int(fc_size)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in tqdm(range(4)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return -correct / total


bounds = [
    {"name": "channels", "type": "discrete", "domain": np.arange(8, 22, 2)},
    {"name": "fc_size", "type": "discrete", "domain": np.arange(10, 105, 5)},
]

bo = GPyOpt.methods.BayesianOptimization(
    f=lambda x: train_and_evaluate(x[:, 0], x[:, 1]),
    domain=bounds,
    acquisition_type="EI",
    exact_feval=True,
    initial_design_numdata=5,
)

bo.run_optimization(max_iter=38)

print(f"Optimal Parameters: Channels={bo.x_opt[0]}, FC Size={bo.x_opt[1]}")

accuracy = -train_and_evaluate(*bo.x_opt)

print(accuracy)
