import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Problem 1
root_dir = '/Users/tildaqvist/Documents/Programmering/DAT565/Assignment_6/'

tensor = torchvision.transforms.ToTensor()
normalize = torchvision.transforms.Normalize((0.5,), (0.5,))
transform = torchvision.transforms.Compose([tensor, normalize])

train_dataset = torchvision.datasets.MNIST(root=root_dir, train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root=root_dir, train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def plot_images(dataset):
    fig, axes = plt.subplots(1, 6, figsize=(12, 3))
    for i, ax in enumerate(axes):
        image, label = dataset[i]
        image = image.squeeze().numpy()
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    plt.show()

plot_images(train_dataset)
plot_images(test_dataset)

# Problem 2
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def create_model_1(image_size, hidden_layer_size, nr_of_classes):
    model = nn.Sequential(
        nn.Linear(image_size, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, nr_of_classes))
    return model

model = create_model_1(image_size=28*28, hidden_layer_size=128, nr_of_classes=10)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images = images.view(-1, 28*28)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy:.4f}')

# Problem 3
def create_model_2(image_size, hidden_layer_size1, hidden_layer_size2, nr_of_classes):
    model = nn.Sequential(
        nn.Linear(image_size, hidden_layer_size1),
        nn.ReLU(),
        nn.Linear(hidden_layer_size1, hidden_layer_size2),
        nn.ReLU(),
        nn.Linear(hidden_layer_size2, nr_of_classes))
    return model

model = create_model_2(image_size=28*28, hidden_layer_size1=500, hidden_layer_size2=300, nr_of_classes=10)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(40):
    model.train()
    for images, labels in train_loader:
        images = images.view(-1, 28*28)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy:.4f}')

# Problem 4
def create_model_3(image_channels, nr_of_classes):
    model = nn.Sequential(
        nn.Conv2d(in_channels=image_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, nr_of_classes))
    return model

model = create_model_3(image_channels=1, nr_of_classes=10)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(40):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy:.4f}')