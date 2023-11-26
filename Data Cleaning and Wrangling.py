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

"""With our dataframe new_1_df containing the information needed, we are ready to move onto cleaning and wrangling the data!
# Data Cleaning and Wrangling
We took a comprehensive approach to data cleaning and wrangling, addressing common data quality issues and preparing the data for subsequent analyses. We achieve this in the following steps:
**Cleaned Dataframe: df_cleaned_V1** size:(170551, 29)
**(1)** Checking NaN distribution among all columns;\
The first step is to identify and handle any missing values (NaNs) across all columns.
**(2)** Padding numerical columns with zeros for all NaN entries;\
We check the NaN distribution across all columns and pad any numerical columns with zeros for NaN entries. This ensures that numerical columns have consistent representations, preventing errors in computations and ensuring effective ML training.
**(3)** Remove rows with NaNs in alphabatical entries;\
For string-based columns, we remove rows with NaNs in alphabetical entries. This helps maintain data integrity and prevent any inconsistencies that could arise from incomplete or inaccurate information.
**(4)** Convert Non-English letters to English letters in alphabatical columns;\
Typically, non-English characters are common in string-based columns. For this reason, we convert any non-English letters to English letters in alphabetical columns. This ensures consistent data representation and facilitates accurate text analysis.
**(5)** Expand ingredients column into several one-hot coding columns padding with 1s and 0s;\
We will transform the "ingredients_text" column into one-hot coding columns, effectively representing the data in a numerical format suitable for ML training. This step allows the ingredients information to be incorporated into the analysis in a structured and interpretable manner.
**(6)** Rename some column names and ready data for following ML training and testing.
We will rename the column names to be more descriptive and meaningful, which will enhance the readability and interpretability of the data. This improves the clarity of the data and facilitates easier understanding for subsequent analyses.

We begin with step (1) by checking the distribution of NaN values across all of the columns.
"""

# Check NaNs value's means
new_1_df.isnull().mean(axis=0)

"""## Proportion of NaN values
The proportion of NaN values can be calculated by getting the mean for NaN values in the dataset. The resulting percentage represents the proportion of data points that are missing for that particular column.
With respect to the Open Food Facts dataset, examining the proportion of NaN values can help us understand the completeness of the information provided for various product attributes. For example, if a significant proportion of values are missing for a column like "ingredients_text", it may indicate that the data collection process needs improvement or that there are limitations in the availability of ingredient information for some products.
Thus, to help us quickly identify columns with high proportions of missing data, we use visualization techniques such bar charts to represent these proportions.
"""

# check the proportion of NaNs in each column
plt.figure(figsize=(5, 10))
new_1_df.isnull().mean(axis=0).plot.barh()
plt.title("Proportion of NaNs in each column")

"""## Why is there missing data?
Upon examining the bar chart above, we can see that several columns exhibit a high proportion of NaN values. These columns include "allergens", "omega-3-fat_100g", "omega-6-fat_100g", "omega-9-fat_100g", "starch_100g", "polyols_100g", and "fruits-vegetables-nuts_100g".
The high proportion of NaN values in the "allergens" column suggests that allergen information is not consistently provided for all products. We believe this is due to variations in product labeling practices or a lack of standardized allergen reporting guidelines. Likewise, the missing values in the omega fatty acid columns "omega-3-fat_100g", "omega-6-fat_100g", and "omega-9-fat_100g" suggests that information about omega fatty acid content is not always available. This could be due to the complexity of measuring omega fatty acids or a lack of emphasis on this nutritional aspect by some manufacturers. Similarly, the absence of starch content data in the "starch_100g" column suggests that this information may not be routinely collected or reported for all food products. Perhaps, there are variations in starch content across different product types or a lack of standardized starch measurement protocols. Overall, we believe there are no standard guidelines when it comes to product labelling practices, which may contribute to the inconsistencies in the dataset.
Thus, the presence of missing data in these columns have many implications for the analysis and interpretation of the Open Food Facts dataset. If we aim to analyze the prevalence of allergens or the nutritional composition of products, the missing values could introduce biases and limit the accuracy of our findings. For this reason, we will drop columns with 100% NaN values.
"""

new_1_df.columns

# drop columns with 100% NaNs
useless_features = ["fruits-vegetables-nuts_100g", "polyols_100g", "starch_100g", "omega-3-fat_100g", "omega-6-fat_100g", "omega-9-fat_100g", "allergens"]
df_cleaned = new_1_df.drop(useless_features, axis=1, inplace=False)

# Sense check new dataframe df_cleaned
print(df_cleaned.shape)
print(df_cleaned.columns)

"""## Replacing NaN values with zeroes appropriately
We assume that NaN values in some columns represent the absence of the corresponding ingredients. We feel this is a reasonable assumption given the context of the dataset. Since the Open Food Facts dataset aims to provide comprehensive information about food products, it is likely that NaN values indicate that the ingredients are not present rather than simply missing due to data collection errors or inconsistencies.
Replacing NaN values with zeroes effectively imputes these missing values by assuming that the ingredient is absent and therefore quantifying its presence as zero. This approach is particularly useful for numerical columns representing quantities of ingredients, such as "fat_100g" or "sugars_100g".
By replacing NaN values with zeroes, we are able to incorporate these columns into our analysis. This allows us to make more informed comparisons between products and identify patterns in ingredient usage. However, we acknowledge that this imputation strategy assumes that the absence of an ingredient is the only possible explanation for a NaN value.
"""

# replace NaNs with zeros in numerical columns
df_cleaned["additives_n"] = df_cleaned["additives_n"].fillna(0)
df_cleaned["energy_100g"] = df_cleaned["energy_100g"].fillna(0)
df_cleaned["fat_100g"] = df_cleaned["fat_100g"].fillna(0)
df_cleaned["saturated-fat_100g"] = df_cleaned["saturated-fat_100g"].fillna(0)
df_cleaned["monounsaturated-fat_100g"] = df_cleaned["monounsaturated-fat_100g"].fillna(0)
df_cleaned["polyunsaturated-fat_100g"] = df_cleaned["polyunsaturated-fat_100g"].fillna(0)
df_cleaned["trans-fat_100g"] = df_cleaned["trans-fat_100g"].fillna(0)
df_cleaned["cholesterol_100g"] = df_cleaned["cholesterol_100g"].fillna(0)
df_cleaned["carbohydrates_100g"] = df_cleaned["carbohydrates_100g"].fillna(0)
df_cleaned["sugars_100g"] = df_cleaned["sugars_100g"].fillna(0)
df_cleaned["fiber_100g"] = df_cleaned["fiber_100g"].fillna(0)
df_cleaned["proteins_100g"] = df_cleaned["proteins_100g"].fillna(0)
df_cleaned["salt_100g"] = df_cleaned["salt_100g"].fillna(0)
df_cleaned["nutrition-score-uk_100g"] = df_cleaned["nutrition-score-uk_100g"].fillna(0)
df_cleaned.head()

# check the proportion of NaNs in each column
plt.figure(figsize=(5, 10))
df_cleaned.isnull().mean(axis=0).plot.barh()
plt.title("Proportion of NaNs in each column")

# drop additives_en column, over 35% data is NaNs
useless_features = ["additives_en"]
df_cleaned = df_cleaned.drop(useless_features, axis=1, inplace=False)

# drop rows containing NaNs
df_cleaned = df_cleaned.dropna()

# check for NaNs, complete.
plt.figure(figsize=(5, 10))
df_cleaned.isnull().mean(axis=0).plot.barh()
plt.title("Proportion of NaNs in each column")

# check for cleaned dataset dimensions
print(df_cleaned.shape)
df_cleaned.head()

!pip install easynmt

# convert all non-english letters into english letters in alphabatical columns
# convert all tags into lower-case, in order for following ML training use
df_cleaned_V1 = df_cleaned.copy()
df_cleaned_V1["product_name"] = df_cleaned_V1["product_name"].str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('utf-8').str.lower()
df_cleaned_V1["brands"] = df_cleaned_V1["brands"].str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('utf-8').str.lower()
df_cleaned_V1["brands_tags"] = df_cleaned_V1["brands_tags"].str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('utf-8').str.lower()
df_cleaned_V1["countries_en"] = df_cleaned_V1["countries_en"].str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('utf-8').str.lower()
df_cleaned_V1["ingredients_text"] = df_cleaned_V1["ingredients_text"].str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('utf-8').str.lower()

df_cleaned_V1.head()

np.sum(df_cleaned_V1["brands"]!=df_cleaned_V1["brands_tags"])

# drop brands_tags column, it's same with brands column
useless_features = ["brands_tags"]
df_cleaned_V1 = df_cleaned_V1.drop(useless_features, axis=1, inplace=False)
df_cleaned_V1.head()

# rename some columns
df_cleaned_V1 = df_cleaned_V1.rename(columns={"countries_en": "countries", "ingredients_text": "ingredients", "additives_n": "additives"})

# explode ingredients column
# add columns indicating whether some ingredients are in this product
# for future ML use, feel free to join more columns here
# list(df_cleaned_V1['ingredients'].str.split(' ', expand=True).stack().unique())
df_cleaned_V1["has_flour"] = df_cleaned_V1["ingredients"].str.contains("flour").apply(lambda x: 1 if x==True else 0)
df_cleaned_V1["has_sugar"] = df_cleaned_V1["ingredients"].str.contains("sugar").apply(lambda x: 1 if x==True else 0)
df_cleaned_V1["has_water"] = df_cleaned_V1["ingredients"].str.contains("water").apply(lambda x: 1 if x==True else 0)
df_cleaned_V1["has_salt"] = df_cleaned_V1["ingredients"].str.contains("salt").apply(lambda x: 1 if x==True else 0)
df_cleaned_V1["has_potassium"] = df_cleaned_V1["ingredients"].str.contains("potassium").apply(lambda x: 1 if x==True else 0)
df_cleaned_V1["has_calcium"] = df_cleaned_V1["ingredients"].str.contains("calcium").apply(lambda x: 1 if x==True else 0)
df_cleaned_V1["has_acid"] = df_cleaned_V1["ingredients"].str.contains("acid").apply(lambda x: 1 if x==True else 0)
df_cleaned_V1["has_alkali"] = df_cleaned_V1["ingredients"].str.contains("alkali").apply(lambda x: 1 if x==True else 0)
df_cleaned_V1["has_iodine"] = df_cleaned_V1["ingredients"].str.contains("iodine").apply(lambda x: 1 if x==True else 0)
df_cleaned_V1["has_milk"] = df_cleaned_V1["ingredients"].str.contains("milk").apply(lambda x: 1 if x==True else 0)
df_cleaned_V1["has_oil"] = df_cleaned_V1["ingredients"].str.contains("oil").apply(lambda x: 1 if x==True else 0)
df_cleaned_V1.head()

# data ready for modeling
print(df_cleaned_V1.shape)
print(df_cleaned_V1.dtypes)
