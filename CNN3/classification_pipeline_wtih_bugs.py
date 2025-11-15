import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from classification_pipeline import Accuracy


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If input and output channels are different, adjust the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)  # Save the input for the residual connection

        # Apply the two convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Add the shortcut (residual connection)
        x += shortcut
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Define a few residual blocks
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)

        # Classifier
        self.fc = nn.Linear(128 * 8 * 8, num_classes)  # Adjust for 32x32 image size

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Training function
def train(model, trainloader, optimizer, criterion, device, accuracy):
    model.train()
    running_loss = 0.0

    log_period = 100
    # log_period = 1 # Overfit on one batch

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        accuracy.update(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % log_period == (log_period - 1):
            print(
                f"[{i + 1}] loss: {running_loss / log_period:.4f}, accuracy: {accuracy.compute()}"
            )
            running_loss = 0.0

            accuracy.reset()

        # break # Overfit on one batch


def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    # Transformations for CIFAR-10 dataset
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4822, 0.4466), (0.2463, 0.2428, 0.2607)),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="../data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="../data", train=False, download=True, transform=transform_test
    )

    # Load CIFAR10 train and test datasets
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=False
    )
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create the ResNet model
    model = ResNet(num_classes=10)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    accuracy = Accuracy()

    for epoch in range(10000):
        train(model, trainloader, optimizer, criterion, device, accuracy)
        evaluate(model, testloader, device)

    # Task:
    # 1. Overfit on one batch
    # Result: loss: 0.0000, accuracy: 1.0

    # 2. Simple train
    # Result:
    # Test accuracy: 78.95%
    # [100] loss: 0.6216, accuracy: 0.7854310274124146
