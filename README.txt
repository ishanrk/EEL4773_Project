EEL 4773 Project

Dependencies Used

- sklearn
- torch
- torchvision
- numpy
- matplotlib
- itertools

How to run the Notebook:
- Sequentially run cells to get the imports, then to load an instance of the tuned model, then to run the augment and train function to train
 the tuned model.
- Finally, run the test function to get an output array of predictions, and accuracy
- A sample version of how to run the code is given in the cell Sample Test

Inputs and Outputs:

a) train function: requires an instance of a model, that we create in the model cell, a numpy array of features x_train, a numpy array of 
  labels called y_train, the number of epochs, and the learning rate (Note: learning rate has a scheduler that multiple it by 0.5 every 5 epochs)
b) test function: requires an instance of a model, that we create in the model cell, and train in the train cell, a numpy arrays of features x_test,
  a numpy array of labels y_train and it returns an array with predictions and the accuracy.
c) ImageCNN model needs input of a list of layers in the following format conv_layers = [{"out_ch":16, "kernel_size": 7, "dropout_rate": 0.3},...] for the number of layers you
 require, and mlp_layers = {"out_ch":256, "dropout_rate":0.5}. However you don't need to specifically do it as we already instantiate a tuned model in the model cell.

Rest of the Code:
The rest of the code is related to hyperparameter tuning and running against other models to get our relevant experiments, it involves a grid search 
and tuning different models and is NOT REQUIRED TO TRAIN OR TEST 