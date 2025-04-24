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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from itertools import product
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay

# other models

def evaluate_model(clf, X_test, y_test):
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds, average='weighted')
    precision = precision_score(y_test, preds, average='weighted')
    return acc, recall, precision

def train_svm_classifier(X_train,y_train, X_test, y_test, pca_components=100):

    pca = PCA(n_components=pca_components)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced  = pca.transform(X_test)


    clf = SVC(kernel='linear', C=1.0, gamma='scale')
    clf.fit(X_train_reduced, y_train)

    return evaluate_model(clf, X_test_reduced, y_test)

def train_knn_with_pca(X_train, y_train, X_test, y_test, pca_components=100):

    pca = PCA(n_components=pca_components)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced  = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_reduced, y_train)
    
    return evaluate_model(knn, X_test_reduced, y_test)

def train_large_mlp_classifier(X_train,y_train, X_test, y_test, hidden_layers=(1024,512, 256, 128, 64)):

    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, activation='relu',
                        solver='adam', max_iter=300, random_state=42)
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
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # lr ← lr * 0.5 every 5 epochs

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


def train(model, x_train, y_train , epochs=25, lr=1e-3, batch_size = 64, device=None):

    X = x_train.reshape(-1, 100, 100)  # reshape to (N, 100, 100)
    X = X / 255.0  # normalize if needed
    X = X[:, np.newaxis, :, :]  # add channel dim: (N, 1, 100, 100)
    #  to tensors
    X_train_tensor = torch.from_numpy(X)
    y_train_tensor = torch.from_numpy(y_train)

    augment = get_augmented_transforms()
    augmented_imgs = torch.stack([augment(img) for img in X_train_tensor])


    X_train_tensor = torch.cat([X_train_tensor, augmented_imgs], dim=0)

    y_train_tensor = torch.cat([y_train_tensor, y_train_tensor], dim=0)

    # Create datasets
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)

    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
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

    x_train = pd.read_csv(feats_csv, header=None).values.astype(np.float32)
    y_train = pd.read_csv(labels_csv, header=None).values.squeeze().astype(np.int64)

    # IF TRAINING FOR THE FINAL TEST
    '''
    model = ImageCNN() 
    history = train(model, feats_csv, labels_csv, epochs=25, lr=1e-2)
    
    
    # IF TRAINING FOR HYPERPARAMETER TUNING
    
    # Set your relevant hyperparaneters in ImageCNN()

    # 1) define the hyperparameter grid
    param_grid = {
        "n_conv":       [2, 3, 4],
        "dropout":      [0.3, 0.5, 0.7],
        "kernel":       [3, 5, 7],
        "out_ch":       [32, 64 ,128],
    }

    out_ch_progressions = {
    2: [(16, 32), (32, 64), (64, 128)],
    3: [(16, 32, 64), (32, 64, 128)],
    4: [(16,32,64,128)]
    # Add more if you support n_conv = 4, etc.
    }
    results = []
    counter = 0
    for n_conv in param_grid["n_conv"]:
        for dropout, kernel, out_chs in product(param_grid["dropout"], param_grid["kernel"], out_ch_progressions[n_conv]):
            conv_layers = [
                {"out_ch": out_chs[i], "kernel_size": kernel, "dropout_rate": dropout}
                for i in range(n_conv)
            ]

            print(f"Training with config: {conv_layers}")
            # build model
            model = ImageCNN(conv_layers=conv_layers, mlp_layers={"out_ch":256, "dropout_rate":0.5})
            # train model
            hist = train_hyperparameter(model, feats_csv, labels_csv, epochs=25, lr=1e-2)
            # save results
            results.append({
                'n_conv': n_conv,
                'dropout':dropout,
                'kernel': kernel,
                'out_ch': out_chs,
                'train_loss': hist['train_loss'],
                'val_loss':   hist['val_loss'],
                'train_acc':  hist['train_acc'],
                'val_acc':    hist['val_acc'],
            })
            print(hist['val_acc'])
            counter += 1
    print(f"Total combinations: {counter}")

    # convert results to DataFrame
    df = pd.DataFrame(results)
    # sort by final validation accuracy
    df['final_val_acc'] = df['val_acc'].apply(lambda x: x[-1])
    df_sorted = df.sort_values('final_val_acc', ascending=False).reset_index(drop=True)

    print("Top 5 configs by validation accuracy:")
    print(df_sorted.head(5)[['n_conv','dropout','kernel','out_ch','final_val_acc']])

    print("Worst 5 configs by validation accuracy:")
    print(df_sorted.tail(5)[['n_conv','dropout','kernel','out_ch','final_val_acc']])

    # save full results
    out_path = 'hyperparam_results.csv'
    df_sorted.to_csv(out_path, index=False)
    print(f"Saved all results to {out_path}")

    # model = ImageCNN() 
    # history = train_hyperparameter(model, feats_csv, labels_csv, epochs=25, lr=1e-2)
    
    '''
    # IF TRAINING TO EXEPRIMENT AGAINST A TEST SET
    conv_layers = [
                {"out_ch":16, "kernel_size": 7, "dropout_rate": 0.3},
        {"out_ch":32, "kernel_size": 7, "dropout_rate": 0.3},
        {"out_ch":64, "kernel_size": 7, "dropout_rate": 0.3},
        {"out_ch":128, "kernel_size": 7, "dropout_rate": 0.3}
                
            ]
    model = ImageCNN(conv_layers=conv_layers, mlp_layers={"out_ch":256, "dropout_rate":0.5}) 
    history,X_train,y_train,X_test,y_test,test_loader = train_experiment(model, feats_csv, labels_csv, epochs=25, lr=1e-2)
    model.eval()
    X_all = []
    y_all = []

    for X_batch, y_batch in test_loader:
        X_all.append(X_batch)
        y_all.append(y_batch)

    X_all = torch.cat(X_all)
    y_all = torch.cat(y_all)

    # Step 2: Move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_all = X_all.to(device)
    y_all = y_all.to(device)

    # Step 3: Forward pass in one go
    model.eval()
    with torch.no_grad():
        outputs = model(X_all)
        preds = outputs.argmax(dim=1)

    # Step 4: Move predictions and labels to CPU for evaluation
    y_true = y_all.cpu().numpy()
    y_pred = preds.cpu().numpy()

    # Step 5: Metrics
    cm = confusion_matrix(y_true, y_pred)

    # Optional: class names (if you have them)
    # Example: class_names = ['cat', 'dog', 'frog']
    num_classes = cm.shape[0]
    class_names = [str(i) for i in range(num_classes)]

    # Create the display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

       # Customize and plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation='horizontal')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    # plt.show()
    plt.savefig('ConfusionMatrix.eps', format='eps')

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    train_loss = history['train_loss']
    train_acc  = history['train_acc']
    epochs = range(1, len(train_loss) + 1)

    # Plot
    # plt.figure(figsize=(10, 4))
    plt.figure()
    # Training Loss
    # plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', color='tab:red')
    plt.plot(epochs, [x / 100 for x in train_acc], label='Train Accuracy', color='tab:blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss and Accuracy')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('TrainingLossAcc.eps', format='eps')


    # # Training Accuracy
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, train_acc, label='Train Accuracy', color='tab:blue')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Training Accuracy')
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

        # When testing against other models just use the test_loader and other data recieved from the above code
    '''
    acc, f1, prec = train_large_mlp_classifier(X_train, y_train, X_test, y_test)
    print(f"RF → Accuracy: {acc:.2f}, F1: {f1:.2f}, Precision: {prec:.2f}")

    acc, f1, prec = train_svm_classifier(X_train, y_train, X_test, y_test)
    print(f"RF → Accuracy: {acc:.2f}, F1: {f1:.2f}, Precision: {prec:.2f}")

    acc, f1, prec = train_knn_with_pca(X_train, y_train, X_test, y_test)
    print(f"RF → Accuracy: {acc:.2f}, F1: {f1:.2f}, Precision: {prec:.2f}")
    '''

    hidden_layers_list = [
    (1024, 512, 256, 128, 64),
    (512, 256, 128, 64),
    (256, 128, 64),
    (1024, 1024, 512, 256, 128, 64),
    (512, 512, 256, 128, 64),
    (256, 256, 128, 64),
    (128, 128, 64),
    (2048, 1024, 512, 256, 128, 64),
    (2048, 2048, 1024, 512, 256, 128, 64),
    ]
    for hidden_layers in hidden_layers_list:
        acc, f1, prec = train_large_mlp_classifier(X_train, y_train, X_test, y_test, hidden_layers=hidden_layers)
        print(f"RF → Accuracy: {acc:.2f}, F1: {f1:.2f}, Precision: {prec:.2f}, Hidden Layers: {hidden_layers}")

    # train those first 
    
    pca_components = [5, 10, 20, 30, 40, 50]
    for n_component in pca_components:
        acc, f1, prec = train_svm_classifier(X_train, y_train, X_test, y_test, pca_components=n_component)
        print(f"RF → Accuracy: {acc:.2f}, F1: {f1:.2f}, Precision: {prec:.2f}, PCA: {n_component}")

        acc, f1, prec = train_knn_with_pca(X_train, y_train, X_test, y_test, pca_components=n_component)
        print(f"RF → Accuracy: {acc:.2f}, F1: {f1:.2f}, Precision: {prec:.2f}, PCA: {n_component}")
    
    
    


if __name__ == '__main__':

    main()