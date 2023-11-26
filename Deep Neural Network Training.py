# For data analysis
import pandas as pd
import numpy as np

# For data visualization
from matplotlib import pyplot as plt
import seaborn as sns

# sklearn libraries
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# torch libraries
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

# Setting the device to do computations on - GPU's are generally faster!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print(device)

#create pytorch dataset and dataloader
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = NN_df

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        features = row[1:]
        label = row[0]
        return features, label

    def __len__(self):
        return len(self.dataframe)

NN_dataset = CustomDataset(NN_df)

#number of rows for 80% of dataset
np.floor(.8*170551)

train_data, test_data = random_split(NN_dataset, [136440, len(NN_dataset) - 136440])

# Batch-size - a hyperparameter
batch = 64
train_loader = DataLoader(train_data, batch_size = batch, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch, shuffle = False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: initialize the neural network layers
        # To flatten your images as vectors so that NN can read them
        # self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(23, 23) #(64*23, 64)
        self.fc2 = nn.Linear(23, 10)
        self.fc3 = nn.Linear(10, 5)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # TODO: implement the operations on input data
        # Hint: think of the neural network architecture for logistic regression
        # , self.softmax
        outputs = nn.Sequential(self.fc1, self.sigmoid, self.fc2, self.sigmoid, self.fc3, self.sigmoid, self.softmax)(x)
        return outputs

LogReg()

# Plotting the Loss
plt.figure(figsize=(10, 5))
plt.plot(range(epoch + 1), loss_LIST_log, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(epoch + 1), acc_LIST_log, label='Training Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()