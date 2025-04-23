import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import StepLR    # learning rate scheduler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

# other models

def evaluate_model(clf, X_test, y_test):
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds, average='weighted')
    precision = precision_score(y_test, preds, average='weighted')
    return acc, recall, precision

def train_svm_classifier(X_train,y_train, X_test, y_test):

    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, y_train)

    return evaluate_model(clf, X_test, y_test)

# DATA LOADING FUNCTIONS

def get_augmented_transforms():
     return transforms.Compose([
         transforms.ToPILImage(),             # expects CxHxW or HxW
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(degrees=15),
         transforms.RandomResizedCrop(size=100, scale=(0.8, 1.0)),
         transforms.ToTensor(),               # gets 1x100x100 for 1 channel
     ])


def load_data_hyperparameters(features_csv, labels_csv, batch_size=64, val_frac=0.2, random_state=42):
    '''
    Use this function while hyperparameter tuning
    It returns a validation loader and a train loader
    mode = "final" indicates it will just return a train loader
    mode = "hyperparameter" indicates it will return a validation and train loader
    mode = "test" indicates it will return a test loader and a train loader
    '''
    X = pd.read_csv(features_csv, header=None).values.astype(np.float32)
    y = pd.read_csv(labels_csv, header=None).values.squeeze().astype(np.int64)

    # reshape to (N, 100, 100), to make sure the images correctly formated
    X = X.reshape(-1, 100, 100)
    # normalize 
    # IMP: CHECK WHETHER THE IMAGE IS ALREADY NORMALIZED    
    # X /= 255.0

    # convert to tensor and add channel dimension: (N, 1, 100, 100)
    X_tensor = torch.from_numpy(X).unsqueeze(1)
    y_tensor = torch.from_numpy(y)

    # create dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # get quick split to get validation and train set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    train_idx, val_idx = next(sss.split(X, y))

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    # dataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_data_experiment(features_csv, labels_csv, batch_size=64, test_frac=0.2, random_state=42):
    '''
    Use this function for a held our test set for final accuracy
    It returns a test loader and a train loader
    '''
    X = pd.read_csv(features_csv, header=None).values.astype(np.float32)
    y = pd.read_csv(labels_csv, header=None).values.squeeze().astype(np.int64)

    X = X.reshape(-1, 100, 100)  # reshape to (N, 100, 100)
    X = X / 255.0  # normalize if needed
    X = X[:, np.newaxis, :, :]  # add channel dim: (N, 1, 100, 100)

    # Initial split: train+val vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    train_idx, test_idx = next(sss1.split(X, y))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]


    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)

    augment = get_augmented_transforms()
    augmented_imgs = torch.stack([augment(img) for img in X_train_tensor])


    X_train_tensor = torch.cat([X_train_tensor, augmented_imgs], dim=0)

    y_train_tensor = torch.cat([y_train_tensor, y_train_tensor], dim=0)

    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)

    # Create datasets
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_data_final(features_csv, labels_csv, batch_size=64):
    '''
    Use this function for returning the full dataset
    '''
    X = pd.read_csv(features_csv, header=None).values.astype(np.float32)
    y = pd.read_csv(labels_csv, header=None).values.squeeze().astype(np.int64)

    X = X.reshape(-1, 100, 100)  # reshape to (N, 100, 100)
    X = X / 255.0  # normalize if needed
    X = X[:, np.newaxis, :, :]  # add channel dim: (N, 1, 100, 100)
    # Convert to tensors
    X_train_tensor = torch.from_numpy(X)
    y_train_tensor = torch.from_numpy(y)

    augment = get_augmented_transforms()
    augmented_imgs = torch.stack([augment(img) for img in X_train_tensor])


    X_train_tensor = torch.cat([X_train_tensor, augmented_imgs], dim=0)

    y_train_tensor = torch.cat([y_train_tensor, y_train_tensor], dim=0)

    # Create datasets
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)

    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    return train_loader


class ImageCNN(nn.Module):
    def __init__(self, num_classes=10, 
                 #conv_filters=[32, 64, 128],
                 #kernel_sizes=[3,3,3], 
                 #dropout_rate=0.5
                 conv_layers = [{"out_ch":32, "kernel_size":5, "dropout_rate":0.5},
                                {"out_ch":64, "kernel_size":5, "dropout_rate":0.5},
                                {"out_ch":128, "kernel_size":5, "dropout_rate":0.5}],
                mlp_layers = {"out_ch":256, "dropout_rate":0.5}
                 ):
        super(ImageCNN, self).__init__()
        layers = []
        in_ch = 1  # single-channel input
        for spec in conv_layers:
            layers += [
                nn.Conv2d(in_ch, spec["out_ch"], kernel_size=spec["kernel_size"], padding='same'),
                nn.BatchNorm2d(spec["out_ch"]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(spec["dropout_rate"]),
            ]
            in_ch = spec["out_ch"]  # update input channels for next layer
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            sample = torch.zeros(1,1,100,100)
            feat = self.conv(sample)
        flat_dim = feat.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat_dim, mlp_layers["out_ch"])
        self.bn1 = nn.BatchNorm1d(mlp_layers["out_ch"])
        self.dropout = nn.Dropout(mlp_layers["dropout_rate"])
        
        self.fc2 = nn.Linear(256, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)                 # (batch, C, H, W)
        x = x.view(x.size(0), -1)        # flatten
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)              # (batch, 256)
        # x = self.mlp(x)                 # (batch, num_classes)
        x = self.fc2(x)                 # (batch, num_classes)
        return x

    # Kaiming weight init
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



# training loop that tracks valid/train loss for plot, uses adam
def train_experiment(model, x_train_csv, y_train_csv , epochs=25, lr=1e-3, device=None):
    train_loader, test_loader = load_data_experiment(x_train_csv, y_train_csv,batch_size=64)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # lr ← lr * 0.1 every 10 epochs

    history = {
        'train_loss': [],
        'train_acc': []
    }

    for epoch in range(1, epochs+1):
        # training
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

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        scheduler.step()  # update learning rate
        print(f"Epoch {epoch}/{epochs} — "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
              #f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%"
              f" — LR: {scheduler.get_last_lr()[0]:.6f}")
    X_train, y_train = [], []
    X_test, y_test = [], []
    for data, labels in train_loader:
        # Flatten each image to a 1D array
        X_train.append(data.view(data.size(0), -1).numpy())
        y_train.append(labels.numpy())
    for data, labels in test_loader:
        # Flatten each image to a 1D array
        X_test.append(data.view(data.size(0), -1).numpy())
        y_test.append(labels.numpy())
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    return history, X_train, y_train, X_test, y_test, test_loader

def train_hyperparameter(model, x_train_csv, y_train_csv , epochs=25, lr=1e-3, device=None):
    train_loader, validation_loader = load_data_hyperparameters(x_train_csv, y_train_csv,batch_size=64)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # lr ← lr * 0.1 every 10 epochs

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(1, epochs+1):
        # training
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

        # valid
        model.eval()
        
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total * 100
        
        # plotting
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step()  # update learning rate
        print(f"Epoch {epoch}/{epochs} — "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
              #f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%"
              f" — LR: {scheduler.get_last_lr()[0]:.6f}")

    return history


def train(model, x_train_csv, y_train_csv , epochs=25, lr=1e-3, device=None):
    train_loader, test_loader = load_data_final(x_train_csv, y_train_csv,batch_size=64)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # lr ← lr * 0.1 every 10 epochs

    history = {
        'train_loss': [],
        'train_acc': []
    }

    for epoch in range(1, epochs+1):
        # training
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

        # plotting
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        scheduler.step()  # update learning rate
        print(f"Epoch {epoch}/{epochs} — "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
              #f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%"
              f" — LR: {scheduler.get_last_lr()[0]:.6f}")
    torch.save(model.state_dict(), 'model_weights.pth')
    return history


def main():
    # data
    feats_csv = 'x_train_project.csv'   
    labels_csv = 't_train_project.csv'

    model = ImageCNN()  

    # IF TRAINING FOR THE FINAL TEST
    '''
    history = train(model, feats_csv, labels_csv, epochs=25, lr=1e-2)
    '''

    # IF TRAINING FOR HYPERPARAMETER TUNING
    '''
    history = train_hyperameter(model, feats_csv, labels_csv, epochs=25, lr=1e-2)
    '''

    # IF TRAINING TO EXEPRIMENT AGAINST A TEST SET
    history,X_train,y_train,X_test,y_test,test_loader = train_experiment(model, feats_csv, labels_csv, epochs=25, lr=1e-2)
    test_correct = 0
    test_total = 0
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == y_batch).sum().item()
                test_total += y_batch.size(0)
            test_acc= test_correct*100 / test_total


    print("Testing Accuracy: ", test_acc)
    # When testing against other models
    '''
    acc, f1, prec = train_random_forest_classifier(X_train, y_train, X_test, y_test, test_loader)
    print(f"RF → Accuracy: {acc:.2f}, F1: {f1:.2f}, Precision: {prec:.2f}")
    '''


if __name__ == '__main__':

    main()

