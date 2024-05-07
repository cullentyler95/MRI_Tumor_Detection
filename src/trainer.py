# train_model.py
import torch
from torch import nn
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Define a simple convolutional neural network architecture
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, (3,3)),  # Convolution layer: 3 input channels, 64 output channels, 3x3 kernel
            nn.ReLU(),                # ReLU activation layer
            nn.Conv2d(64, 256, (3,3)),# Second convolution layer with increased depth
            nn.ReLU(),                # ReLU activation layer
            nn.Conv2d(256, 64, (3,3)),# Third convolution layer reducing depth
            nn.ReLU(),                # ReLU activation layer
            nn.Flatten(),             # Flatten the output for linear layers
            nn.Linear(64*(224-6)*(224-6), 64),  # Dense layer after flattening
            nn.Linear(64, 64),        # Additional dense layer with 64 units
            nn.Linear(64, 2)          # Final output layer with 2 units for classification
        )

    def forward(self, x):
        # Forward pass through the network
        return self.model(x)

def create_data_loaders(data_path, batch_size=4):
    # Apply transformations and load images from directory
    transforms = Compose([ToTensor(), Resize((224, 224))])
    dataset = ImageFolder(data_path, transform=transforms)
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    # DataLoader objects to handle data batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

def train(device, train_loader, valid_loader, epochs=10):
    clf = Classifier().to(device)
    opt = Adam(clf.parameters(), lr=1e-5)  # Optimizer
    loss_fn = nn.CrossEntropyLoss()  # Loss function for classification
    clf.train()  # Set model to training mode
    train_losses = []
    train_accs = []
    valid_accs = []
    for epoch in range(epochs):
        correct = 0.0
        items = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            yhat = clf(X)
            loss = loss_fn(yhat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pred = torch.argmax(yhat, 1)
            correct += (y == pred).sum().item()
            items += y.size(0)
        # Calculate accuracy for this epoch and store losses, train_accs
        train_acc = correct / items
        train_losses.append(loss.item())
        train_accs.append(train_acc * 100)

        # Validate
        clf.eval()
        with torch.no_grad():
            valid_correct = 0.0
            valid_items = 0.0
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                yhat = clf(X)
                pred = torch.argmax(yhat, 1)
                valid_correct += (y == pred).sum().item()
                valid_items += y.size(0)
            valid_acc = valid_correct / valid_items
            valid_accs.append(valid_acc * 100)
        # Switch back to train mode
        clf.train()

        print(f"Epoch {epoch} loss {loss.item()} train_acc {train_acc * 100} valid_acc {valid_acc * 100}")

    # After training, plot Loss and Accuracy for Training and Validation Sets
    epochs = range(1, epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train')
    plt.plot(epochs, valid_accs, label='Valid')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    return clf

def save_model(model, model_path):
    # Save the model state dictionary to a file
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = './data'
    train_loader, valid_loader = create_data_loaders(data_path)
    model = train(device, train_loader, valid_loader)
    model_path = './TumorModel.pth'
    save_model(model, model_path)