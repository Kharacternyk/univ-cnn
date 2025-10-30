import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def custom_gelu(x):
    """Custom GELU activation function."""
    return x * 0.5 * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)))

def cumulative_distribution_function(x, N=10):
    return (torch.randn((N, *x.shape), device=x.device) < x).sum(0) / N

def gelu_with_random(x):
    return x #* cumulative_distribution_function(x)

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)

    def forward(self, x):
        x = self.pool(gelu_with_random(self.conv1(x)))
        x = self.pool(gelu_with_random(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the feature maps
        x = gelu_with_random(self.fc1(x))
        x = self.fc2(x)  # No softmax, because CrossEntropyLoss applies it automatically
        return x



if __name__ == '__main__':
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean & std of MNIST
    ])

    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNIST_CNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    num_epochs = 5  # Set number of training epochs

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # No gradients needed for inference
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)  # Get class with highest probability
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

    print("Training complete!")


# Epoch [1/5], Loss: 0.1361
# Test Accuracy: 98.48%
# Epoch [2/5], Loss: 0.0426
# Test Accuracy: 98.62%
# Epoch [3/5], Loss: 0.0287
# Test Accuracy: 99.06%
# Epoch [4/5], Loss: 0.0207
# Test Accuracy: 98.86%
# Epoch [5/5], Loss: 0.0161
# Test Accuracy: 99.08%
# Training complete!