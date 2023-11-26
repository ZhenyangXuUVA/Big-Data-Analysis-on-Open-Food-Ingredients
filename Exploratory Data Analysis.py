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

# drive.flush_and_unmount()
drive.mount('/content/drive')
df = pd.read_table('/content/drive/MyDrive/en.openfoodfacts.org.products.tsv')
#df = pd.read_table('/content/drive/MyDrive/5450/en.openfoodfacts.org.products.tsv')
# df = pd.read_table('/content/drive/MyDrive/kaggle/en.openfoodfacts.org.products.tsv')

"""# Exploratory Data Analysis
The Open Food Facts dataset is quite large with a size of about 1 GB (~360000 entries!). So, it would make sense to first conduct EDA to understand what we are working with. To have an idea, let us take a look at the first 10 rows of our dataframe df.
"""

# Peek first 10 rows of the data set
df.head(10)

"""## What are the data types?
From our initial observations, we can see that the dataframe contains information on a variety of food products, including soft drinks, teas, and other beverages. The products are from a variety of countries, including the United States, Germany, Spain, the United Kingdom, and France. The dataframe also includes information on the nutrients in each product, such as energy, sugar, and fat per 100 grams. Likewise, there are columns that list the ingredients and allergens (if applicable) for each product. On the other hand, there are plenty of columns that the Open Food Facts website uses as part of their backend code such as when a product entry was created ("created_datetime") or when it was last modified ("last_modified_datetime"). These columns are not particularly useful to us. However, columns like("product_name") and ("ingredients_text") will definitely be of interest.
Next, we shall take a look at what sort of data types we are dealing with in the columns of this dataframe.
"""

# Check data types of columns and the shape of dataframe
df.dtypes

"""## What is the shape?
We can observe that the data types are quite diverse, consisting of objects and float64. Since a majority of the columns contain object values, this indicates that the dataset incorporates a variety of data formats, including strings (product name, categorie, country), numerical data (nutrients per 100 grams), and even comma-separated values (ingredients).
The presence of comma-separated values within the "ingredients_text" column suggests that the nutritional information for each product is some what structured and organized, allowing us the potential to analyze and explore nutritional patterns and trends. As a result, this means we will need to cast the data types appropriately so that it can be parsed.
Now, we will take a look at the shape of this dataframe and the descriptive summary.
"""

# Get the shape of the dataframe
df.shape

"""## A descriptive summary of our dataframe
As alluded to earlier in our introduction, the Open Food Facts dataset is quite large and comprehensive with a shape that is an astonishing 356027 rows and 163 columns! This tells us that the dataset has a significant amount of information on a wide range of food products. The large number of rows implies that the dataset can potentially provide detailed analysis and insights into the various aspects of food products, such as their origins, pricing, and nutritional composition. Likewise, the large number of columns suggests that the dataset covers a broad range of attributes related to each food product.

Then, we shall take a look at the descriptive summary of this dataframe.
"""

# Get a descriptive summary of the dataframe
df.describe()

'''
selected_feature = 'nutrition-score-fr_100g'
df['nutrition-score-fr_100g'].value_counts().plot(kind='bar')
plt.title(f'Distribution of {selected_feature}')
plt.xlabel(selected_feature)
plt.ylabel('Count')
plt.show()
'''

"""## What columns do we have?
Since there are 163 columns in the dataframe, there is no way we can display them all at once when viewing the dataframe. So, one way is to iterate all the columns in the dataframe and print them out.
"""

# Look through all the available columns in the dataset
for col in df.columns:
  print(col)

"""## Selecting relevant columns
Selecting relevant columns is a crucial step to ensure that the analysis focuses on the most informative and pertinent data. Given our logistic regression and clustering tasks, we are primarily interested in the ingredients of the food products, as these provide insights into the nutritional properties and potential health implications of the products. Therefore, we are dropping columns that are not useful for the tasks such as data entry and packaging information; only keeping ingredients-related columns.
By focusing on ingredients-related columns, we can effectively streamline the analysis process, reducing computational complexity and improving the efficiency of our models. Moreover, excluding irrelevant columns helps to minimize noise and potential biases that may arise from unrelated data, enhancing the accuracy and reliability of our results.
We will create a new dataframe called new_df by selecting the columns we want. Then, we will cast the columns containing objects to string data type.
"""

# Select relevant columns
new_df = df[['product_name', 'brands','brands_tags', 'countries_en', 'ingredients_text', 'allergens', 'additives_n', 'additives_en', 'energy_100g', 'fat_100g', 'saturated-fat_100g', 'monounsaturated-fat_100g', 'polyunsaturated-fat_100g', 'omega-3-fat_100g', 'omega-6-fat_100g', 'omega-9-fat_100g', 'trans-fat_100g', 'cholesterol_100g', 'carbohydrates_100g', 'sugars_100g', 'starch_100g', 'polyols_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'fruits-vegetables-nuts_100g', 'nutrition-score-uk_100g']]

# Cast types of columns and confirm change
new_df['product_name'] = new_df['product_name'].astype('string')
new_df['brands'] = new_df['brands'].astype('string')
new_df['brands_tags'] = new_df['brands_tags'].astype('string')
new_df['countries_en'] = new_df['countries_en'].astype('string')
new_df['ingredients_text'] = new_df['ingredients_text'].astype('string')
new_df['allergens'] = new_df['allergens'].astype('string')
new_df['additives_en'] = new_df['additives_en'].astype('string')
new_df = new_df.sort_values(by='additives_n', ascending=False)

"""We can now take a quick look at the result of our new dataframe."""
# Check the data types for the columns
new_df.dtypes

# Sense check cleaned df
new_df.info()

"""At this point, it makes sense to take a look at the first 10 rows of our dataframe new_df to get a sense of how things look. We have not dropped any null values yet, so it is expected that there will be some rows with this problem."""
# Print the first 10 rows of new_df
new_df.head(10)

"""To limit ourselves on English data, we narrow down the dataset to only entries from the United States and where the ingredient list is not empty."""
# Filter the rows for products that are in the United States
new_1_df = new_df[new_df['countries_en'] == 'United States']

# Drop any rows that contain null as the ingredients
new_1_df.dropna(subset=['ingredients_text'], inplace=True)

# Display the data frame new_1_df
new_1_df

