import numpy as np
import pandas as pd
from train import ImageCNN
import torch
def test(x_test_csv, y_test_csv):
    '''
    Takes in two csv files for features: x_test_csv
    and for labels: y_test_csv
    and outputs a vector of predictions
    '''
    X = pd.read_csv(x_test_csv, header=None).values.astype(np.float32)
    y = pd.read_csv(y_test_csv, header=None).values.squeeze().astype(np.int64)

    # reshape to (N, 100, 100), to make sure the images correctly formated
    X = X.reshape(-1, 100, 100)
    # normalize 
    # IMP: CHECK WHETHER THE IMAGE IS ALREADY NORMALIZED    
    # X /= 255.0

    # convert to tensor and add channel dimension: (N, 1, 100, 100)

    X_tensor = torch.from_numpy(X).unsqueeze(1)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model = ImageCNN()

    model.load_state_dict(torch.load('model_weights.pth'))

    model.eval()

    preds = model(X_tensor.to(device=device))

    preds =  preds.argmax(dim=1)

    return preds



def test_accruacy(predictions, y_test_array):
    '''
    Takes in a vector (np array) of predictions
    and a vector (np array) of ground truths y_test_array
    and returns accuracy
    '''
    return np.sum(predictions==y_test_array)/ len(y_test_array)