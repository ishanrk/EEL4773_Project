import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

# 1. Load data function
def load_data(features_csv, labels_csv, batch_size=64, val_frac=0.2, random_state=42):
    # Read CSVs: assume each row is a flattened 100x100 image (length 10000)
    X = pd.read_csv(features_csv, header=None).values.astype(np.float32)
    y = pd.read_csv(labels_csv, header=None).values.squeeze().astype(np.int64)

    # Reshape to (N, 100, 100)
    X = X.reshape(-1, 100, 100)
    # Normalize pixel values to [0,1]
    X /= 255.0

    # Convert to tensor and add channel dimension: (N, 1, 100, 100)
    X_tensor = torch.from_numpy(X).unsqueeze(1)
    y_tensor = torch.from_numpy(y)

    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    train_idx, val_idx = next(sss.split(X, y))

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# 2. Data augmentation transforms (for on-the-fly augmenting within a custom Dataset)
def get_augmented_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),             # expects CxHxW or HxW
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=100, scale=(0.8, 1.0)),
        transforms.ToTensor(),               # yields 1x100x100 for grayscale
    ])

# 3. CNN model for single-channel 100x100 input, 10 classes
class ImageCNN(nn.Module):
    def __init__(self, num_classes=10, conv_filters=[32, 64, 128],
                 kernel_sizes=[3,3,3], dropout_rate=0.5):
        super(ImageCNN, self).__init__()
        layers = []
        in_ch = 1  # single-channel input
        for out_ch, k in zip(conv_filters, kernel_sizes):
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout_rate)
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # Compute flattened feature size dynamically
        with torch.no_grad():
            sample = torch.zeros(1,1,100,100)
            feat = self.conv(sample)
        flat_dim = feat.view(1, -1).size(1)

        # Fully connected head
        self.fc1 = nn.Linear(flat_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)                 # (batch, C, H, W)
        x = x.view(x.size(0), -1)        # flatten
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

# 4. Training loop function (returns history for plotting)
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(1, epochs+1):
        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        avg_train_loss = train_loss / total
        train_acc = correct / total * 100

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total * 100

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{epochs} — "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")

    return history

# 5. Example usage
def main():
    # Paths to your CSVs
    feats_csv = 'x_train_project.csv'
    labels_csv = 't_train_project.csv'

    # Load data
    train_loader, val_loader = load_data(feats_csv, labels_csv)

    # Instantiate model
    model = ImageCNN()

    # Train and capture history
    history = train_model(model, train_loader, val_loader, epochs=10, lr=1e-3)

    # Plotting final accuracy and loss
    import matplotlib.pyplot as plt
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.show()

    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.show()

if __name__ == '__main__':
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    
    main()

