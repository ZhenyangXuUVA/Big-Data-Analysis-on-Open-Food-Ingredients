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

"""#Isolating Numerics from Categorical Features
Create groups of the numeric and categorical variables
numerics_df: This dataframe contains all numerical columns in df_cleaned_V1 to be used in Linear Regression
one_hot_df: This dataframe contains all columns of type 'int64'; one-hot encoded columns created earlier to be used for clustering
categorical_df: This dataframe contains all categorical columns from df_cleaned_V1
"""

numerics_df = df_cleaned_V1.select_dtypes(include = 'number')
categorical_df = df_cleaned_V1.select_dtypes(include = 'object')

"""Next, we check the correlations among all the columns in numerics_df using correlation heatmap.
First, we create a correlation matrix using numerics_df and call it corr_mat. Using the correlation matrix, we generated a correlation heatmap for these numeric features using Seaborn library.
"""

#create correlation matrix
corr_mat = numerics_df.corr()

#Create plot
##set figsize
fig, ax = plt.subplots(figsize=(6, 6))
##create heatmap
sns.heatmap(corr_mat, cmap = 'RdBu', vmin = -1, vmax = 1, center = 0)
plt.title('Correlation Heatmap of numerical variables')

plt.show()

"""Here we see that fat_100g and other fats (saturated, monosaturated and polysaturated) as highly correlated because the former is a parent class. Thus we will be excluding fat_100g in subsequent modeling tasks."""

numerics_df = numerics_df.drop('fat_100g', axis = 1)

numerics_df.columns

"""# Model Building and Data Analysis"""

numerics_df['nutrition-score-uk_100g'].describe()

#Based on Nutri-Score formula, food are categorized in 5 categories.
df_cleaned_V1['label'] = pd.cut(x = df_cleaned_V1['nutrition-score-uk_100g'], bins=[-15, 0, 3, 11, 19, 40], labels = ['A','B','C','D','E'])

##Linear
##K-means clustering

#Comparison of different models
##NN
##random forest decision tree

#measuring of performance
##10 classes -
##precision & recall

#findings and conclusions along the way

"""Now that we have explored and cleaned our dataset, here we create Features and Labels. We also split the data into Train and Test sets."""

features = numerics_df.drop('nutrition-score-uk_100g', axis = 1)

target = numerics_df['nutrition-score-uk_100g']

seed = 42
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=seed)

"""##Linear Regression
Here we are using the features defined earlier to predict the nutrition-score-uk_100g.
"""

#Initialize model with default parameters and fit it on the training set
reg = LinearRegression()
reg.fit(X_train, y_train)

# TO-DO: Use the model to predict on the test set and save these predictions as `y_pred`
y_pred = reg.predict(X_test)

# TO-DO: Find the R-squared score and store the value in `lin_reg_score`
lin_reg_score = sklearn.metrics.r2_score(y_test, y_pred)

lin_reg_score

"""From the R2 score, we can see that the features are able to explain about 61% of the variance in the UK nutrition score."""

residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--', label="Residuals Mean")
plt.title("Residual Plot")
plt.legend()
plt.show()

"""##Logistic Regression
Here we are predicting the UK Nutrition categories using the features.
"""

features = numerics_df.drop('nutrition-score-uk_100g', axis = 1)
target = df_cleaned_V1['label']

seed = 42
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=seed)

# TO-DO: Initialize model with default parameters and fit it on the training set
clf = LogisticRegression()
clf.fit(X_train,y_train)

# TO-DO: Use the model to predict on the test set and save these predictions as `y_pred`
y_pred = clf.predict(X_test)

# TO-DO: Find the accuracy and store the value in `log_acc`
log_acc = sklearn.metrics.accuracy_score(y_pred,y_test)
print("Accuracy: %.1f%%"% (log_acc*100))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""##Logistic Regression with PCA
We want to reduce noise by using PCA to improve logistic regression.
"""

len(features.columns)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Instantiate and Fit PCA
pca = PCA()
pca.fit(X_train_scaled)
pca.n_components_

# TO-DO: Save the explained variance ratios into variable called "explained_variance_ratios"
explained_variance_ratios = pca.explained_variance_ratio_

# TO-DO: Save the CUMULATIVE explained variance ratios into variable called "cum_evr"
cum_evr = np.cumsum(pca.explained_variance_ratio_)

# TO-DO: find optimal num components to use (n) by plotting explained variance ratio (2 points)
#set figure size
fig, ax = plt.subplots(figsize=(8, 6))
#set x-labels
x_labels = [i for i in range(1,31)]

#draw
sns.lineplot(
    data = cum_evr
)
plt.axhline(0.8, color = 'red')
plt.axvline(12)

ax.set_xlabel('Number of components')
ax.set_ylabel('Variance Explained')
ax.set_title('Explained variance ratio')
ax.set_xticks([i for i in range(0,30)],x_labels)

plt.show()

"""We see that using 13 components is able to explain 80% of the variance.
Below, we train and fit PCA using 13 components.
"""

# TO-DO: Get transformed set of principal components on x_test

# 1. Refit and transform on training with parameter n (as deduced from the last step)
pca2 = PCA(n_components=13)

#pca.fit(X_train)
X_train_pca = pca2.fit_transform(X_train_scaled)

# 2. Transform on Testing Set and store it as `X_test_pca`
X_test_pca = pca2.transform(X_test_scaled)

# TO-DO: Initialize `log_reg_pca` model with default parameters and fit it on the PCA transformed training set
log_reg_pca = LogisticRegression()
log_reg_pca.fit(X_train_pca, y_train)

# TO-DO: Use the model to predict on the PCA transformed test set and save these predictions as `y_pred`
y_pred = log_reg_pca.predict(X_test_pca)

# TO-DO: Find the accuracy and store the value in `test_accuracy`
test_accuracy = sklearn.metrics.accuracy_score(y_pred,y_test)

test_accuracy

"""We see that the accuracy score has improved from 49.9% to 55.6% using PCA."""

plt.bar(range(1, len(pca2.explained_variance_ratio_) + 1), pca2.explained_variance_ratio_)
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance Ratio of Principal Components")
plt.show()

"""Drop One-hot columns with PCA"""

df_cleaned_V1.columns

features = df_cleaned_V1[['additives', 'energy_100g', 'saturated-fat_100g',
       'monounsaturated-fat_100g', 'polyunsaturated-fat_100g',
       'trans-fat_100g', 'cholesterol_100g', 'carbohydrates_100g',
       'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g']]

seed = 42
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=seed)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Instantiate and Fit PCA
pca = PCA()
pca.fit(X_train_scaled)
pca.n_components_

# TO-DO: Save the explained variance ratios into variable called "explained_variance_ratios"
explained_variance_ratios = pca.explained_variance_ratio_

# TO-DO: Save the CUMULATIVE explained variance ratios into variable called "cum_evr"
cum_evr = np.cumsum(pca.explained_variance_ratio_)

# TO-DO: find optimal num components to use (n) by plotting explained variance ratio (2 points)
#set figure size
fig, ax = plt.subplots(figsize=(8, 6))
#set x-labels
x_labels = [i for i in range(1,12)]

#draw
sns.lineplot(
    data = cum_evr
)
plt.axhline(0.8, color = 'red')
plt.axvline(6)

ax.set_xlabel('Number of components')
ax.set_ylabel('Variance Explained')
ax.set_title('Explained variance ratio')
ax.set_xticks([i for i in range(0,11)],x_labels)

plt.show()

# TO-DO: Get transformed set of principal components on x_test

# 1. Refit and transform on training with parameter n (as deduced from the last step)
pca2 = PCA(n_components=7)

#pca.fit(X_train)
X_train_pca = pca2.fit_transform(X_train_scaled)

# 2. Transform on Testing Set and store it as `X_test_pca`
X_test_pca = pca2.transform(X_test_scaled)

# TO-DO: Initialize `log_reg_pca` model with default parameters and fit it on the PCA transformed training set
log_reg_pca = LogisticRegression()
log_reg_pca.fit(X_train_pca, y_train)

# TO-DO: Use the model to predict on the PCA transformed test set and save these predictions as `y_pred`
y_pred = log_reg_pca.predict(X_test_pca)

# TO-DO: Find the accuracy and store the value in `test_accuracy`
test_accuracy = sklearn.metrics.accuracy_score(y_pred,y_test)
test_accuracy

cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels = ['A','B','C','D','E']))

cm

#visualizing the confusion matrix
plt.figure(figsize = (8,4))
ax = sns.heatmap(cm, annot=True, annot_kws={"size": 8}, square=True, fmt = 'g', cmap = 'GnBu')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

features = numerics_df.drop('nutrition-score-uk_100g', axis = 1)
target = df_cleaned_V1['label']

"""We see that the distribution of class labels is not even. We need so do resample."""

df_cleaned_V1.groupby('label')['product_name'].count()

"""Resampling the training data and train the model"""

ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(X_train, y_train)

#scale the data
scaler2 = StandardScaler()

X_ros_train_scaled = scaler2.fit_transform(X_ros)

X_ros_test_scaled = scaler2.transform(X_test)

# TO-DO: Get transformed set of principal components on x_test

# 1. Refit and transform on training with parameter n (as deduced from the last step)
pca2 = PCA(n_components=7)

#pca.fit(X_train)
X_ros_train_pca = pca2.fit_transform(X_ros_train_scaled)

# 2. Transform on Testing Set and store it as `X_test_pca`
X_ros_test_pca = pca2.transform(X_ros_test_scaled)

# TO-DO: Initialize `log_reg_pca` model with default parameters and fit it on the PCA transformed training set
log_reg_pca2 = LogisticRegression()
log_reg_pca2.fit(X_ros_train_pca, y_ros)

# TO-DO: Use the model to predict on the PCA transformed test set and save these predictions as `y_pred`
y_pred = log_reg_pca.predict(X_ros_test_pca)

# TO-DO: Find the accuracy and store the value in `test_accuracy`
test_accuracy = sklearn.metrics.accuracy_score(y_pred,y_test)
test_accuracy

cm2 = pd.DataFrame(confusion_matrix(y_test, y_pred, labels = ['A','B','C','D','E']))

#visualizing the confusion matrix
plt.figure(figsize = (8,4))
ax = sns.heatmap(cm2, annot=True, annot_kws={"size": 8}, square=True, fmt = 'g', cmap = 'GnBu')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

"""Neural Network"""

df_cleaned_V1.head()

target

NN_df = df_cleaned_V1.select_dtypes(include = 'number').drop(['fat_100g', 'nutrition-score-uk_100g'],axis=1)

df_cleaned_V1['numeric_target'] = pd.cut(x = df_cleaned_V1['nutrition-score-uk_100g'], bins=[-15, 0, 3, 11, 19, 40], labels = [0,1,2,3,4])

df_cleaned_V1.astype({'numeric_target': 'int64'}).dtypes

NN_df.insert(0, "label", df_cleaned_V1['numeric_target'])

NN_df = NN_df.astype(np.float32)

NN_df.dtypes

NN_df = NN_df.astype({'label': 'float32'}) #int32

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

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Sending the data to device (CPU or GPU)
# # TODO: (1 of 2)
# # Step 1: instantiate the logistic regression to variable logreg
# logreg = LogReg().to(device)
# 
# # Step 2: set the loss criterion as CrossEntropyLoss
# criterion = nn.CrossEntropyLoss()
# 
# # END TODO
# optimizer = optim.Adam(logreg.parameters(), lr=1e-3) #lr - learning step
# # optimizer = optim.SGD(logreg.parameters(), lr=1e-5, momentum=0.9)
# epoch = 50
# 
# loss_LIST_log = []
# acc_LIST_log = []
# 
# # Train the Logistic Regression
# for epoch in range(epoch):
#   running_loss = 0.0
#   correct = 0
#   total = 0
#   for inputs, labels in train_loader:
#       labels = labels.type(torch.LongTensor) # Cast to Float
#       inputs, labels = inputs.to(device), labels.to(device)
# 
#       ## TODO (2 of 2)
#       # Step 1: Reset the optimizer tensor gradient every mini-batch
#       optimizer.zero_grad()
# 
#       # Step 2: Feed the network the train data
#       outputs = logreg(inputs)
# 
#       # Step 3: Get the prediction using argmax
#       preds = torch.argmax(outputs, axis=1)
# 
#       # Step 4: Find average loss for one mini-batch of inputs
#       loss = criterion(outputs, labels)
# 
#       # Step 5: Do a back propagation
#       loss.backward()
# 
#       # Step 6: Update the weight using the gradients from back propagation by learning step
#       optimizer.step()
# 
#       # Step 7: Get loss and add to accumulated loss for each epoch
#       running_loss += loss.item() * len(labels)
# 
#       # Step 8: Get number of correct prediction and increment the number of correct and total predictions after this batch
#       # Hint: we need to detach the numbers from GPU to CPU, which stores accuracy and loss
#       correct += (preds == labels).sum().item()
#       total += len(preds)
# 
#   # Step 9: Calculate training accuracy for each epoch (should multiply by 100 to get percentage), store in variable called 'accuracy', and add to acc_LIST_log
#   accuracy = correct / len(train_data) * 100
#   acc_LIST_log.append(accuracy)
# 
#   # Step 10: Get average loss for each epoch and add to loss_LIST_log
#   avg_loss = running_loss / len(train_data)
#   loss_LIST_log.append(avg_loss)
# 
#   # print statistics
#   print("The loss for Epoch {} is: {}, Accuracy = {}".format(epoch, running_loss/len(train_loader), accuracy))
#

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