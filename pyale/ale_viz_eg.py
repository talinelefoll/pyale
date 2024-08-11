
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os

from ale_1d_calcu import calculate_ale_1d
from ale_1d_calcu import ale_1d_plot

# Load data from CSV
# Relative path to the dataset from the script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
data_path = os.path.join(base_dir, '../tests/var_cars.csv')  # Going one level up to the 'tests' directory
data = pd.read_csv(data_path)


# List of categorical columns
categorical_cols = ["vs", "am", "gear", "country"]

# Apply Label Encoding
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# convert into categorical datatype
data = data.astype({'vs': 'category', 'am': 'category','gear':'category','country': 'category'})

# load dataset into a DataFrame 'data'
X = data.drop(columns=['mpg'])  # Features
y = data['mpg']  # Target variable

# Optionally, split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_1 = LinearRegression()
# Train the model
model_1.fit(X_train, y_train)

# Example evaluation
train_score = model_1.score(X_train, y_train)
test_score = model_1.score(X_test, y_test)

# Calculate ale_1d results
ale_1d_calcul = calculate_ale_1d('LinearRegression', model_1, X, y, n_bootstrap=2, subsample=10000, n_jobs=1, n_bins=20)

        
# Example dataset setup
# Assuming `data` is already defined and loaded as a DataFrame and `ale_1d_ds` is your ALE results

# Ensure the columns are properly categorized
data = data.astype({
    'vs': pd.CategoricalDtype(categories=['FALSE', 'TRUE']),
    'am': pd.CategoricalDtype(categories=['TRUE', 'FALSE']),
    'gear': pd.CategoricalDtype(categories=['three', 'four', 'five'], ordered=True),
    'country': pd.CategoricalDtype(categories=['Sweden', 'UK', 'Japan', 'Italy', 'Germany', 'USA'])
})

# Usage example
ale_1d_plot(data, target_col='mpg', ale_ds=ale_1d_ds, categorical_cols=['vs', 'am', 'gear', 'country'], centralization='median')
# ale_plot(data, target_col='mpg', ale_ds=ale_1d_ds, categorical_cols=['vs', 'am', 'gear', 'country'], centralization='mean')
# ale_plot(data, target_col='mpg', ale_ds=ale_1d_ds, categorical_cols=['vs', 'am', 'gear', 'country'], centralization='zero')