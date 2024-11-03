import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from time import time
from tqdm import trange



# Load and Normalize CIFAR-10 Data

trans_cifar10_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

trainset = torchvision.datasets.CIFAR10(root='data/cifar10', train=True,
                                        download=True, transform=trans_cifar10_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='data/cifar10', train=False,
                                       download=True, transform=trans_cifar10_val)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# CNNCifar Model
class CNNCifar(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        feature = F.relu(self.fc2(x))
        x = self.fc3(feature)
        return F.log_softmax(x, dim=1), feature

num_classes = 10

print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = CNNCifar(num_classes).to(device)
print('Device: ', device)

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

# Function to calculate accuracy
def calculate_accuracy(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training and Testing over Epochs
training_accuracies = []
testing_accuracies = []

for epoch in trange(1500, desc = "Training:"):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs, _ = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Calculate Training and Testing Accuracy
    train_acc = calculate_accuracy(trainloader)
    test_acc = calculate_accuracy(testloader)
    training_accuracies.append(train_acc)
    testing_accuracies.append(test_acc)
    
    print(f'Epoch {epoch+1}, Loss: {running_loss:.4f}, Training Acc: {train_acc:.2f}, Testing Acc: {test_acc:.2f}')

# Save accuracies to files
with open('training_accuracies.txt', 'w') as f:
    for acc in training_accuracies:
        f.write(f"{acc}\n")

with open('testing_accuracies.txt', 'w') as f:
    for acc in testing_accuracies:
        f.write(f"{acc}\n")

print('Finished Training. Accuracies saved to files.')
