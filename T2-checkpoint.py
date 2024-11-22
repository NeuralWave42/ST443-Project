# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:15:07 2024

@author: emmaqueen
"""

#%% IMPORT THE LIBRARIES
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
 import sklearn.linear_model as skl


#%% IMPORT THE DATA
os.chdir('C:/Users/emmaqueen/Documents/ST443/PROJECT/')
current_directory = os.getcwd()

file2 = pd.read_csv('data2.csv.gz')
print(file2.head())
#pd.set_option("display.precision", 2)
#%%
#Explore the data to generate summary statistics and plots that help the reader un
#derstand the data, with a focus on information relevant to the classification task.

shape = file2.shape
info = file2.info()
describe = file2.describe
counts_of_actives_cells = file2["label"].value_counts()
missing_values= file2.isnull().sum().sum()
atoms_columns = file2.columns  # Get column names
sparcity = (file2 == 0).mean().mean()  # Calculate sparsity



print("The number od rows and columns is : ", shape)
print(" The statistic summary is : ", describe)
print("The percentage of active cells is :", counts_of_actives_cells)
print("Missing values in total in the Data set : ", missing_values)
print(f"Sparsity: {sparcity:.2f}")
#%%

# PLOT OF THE INACTIVE AGAINST ACTIVE COMPOUNDS
file2['label'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.title("Distribution of Binding (label=1) vs. Non-Binding (label=-1) Compounds")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()




#%%
# Load the dataset
# Assuming file2 is already loaded with 'label' as the target column

X = file2.iloc[:, 1:]  # Features
y = file2.iloc[:, 0]   # Target (label)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =============================================================================
# Method 1: Lasso
# =============================================================================
#%%

#%% IMPORT THE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier

#%% IMPORT THE DATA
file_path = 'data2.csv.gz'  # Replace with your actual file path
file2 = pd.read_csv(file_path)

# Separate features and target
X = file2.iloc[:, 1:]  # Features
y = file2.iloc[:, 0]   # Target (label)

# Check sparsity
sparcity = (X == 0).mean().mean()
print(f"Sparsity of the dataset: {sparcity:.2f}")

#%% Data Preprocessing
# Standardize features (important for Lasso)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%% Lasso Regression for Feature Selection
# Use LassoCV to automatically tune the regularization parameter alpha
lasso = LassoCV(cv=5, random_state=42, n_alphas=100)
lasso.fit(X_scaled, y)

# Identify selected features
selected_features = np.where(lasso.coef_ != 0)[0]  # Indices of selected features
print(f"Number of selected features: {len(selected_features)}")

# Reduce dataset to selected features
X_selected = X_scaled[:, selected_features]

#%% Train/Test Split with Selected Features
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Train a simple classifier to evaluate the selected features
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = rf_classifier.predict(X_test)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy with selected features: {balanced_acc:.4f}")

#%% Visualizing Lasso Coefficients
plt.figure(figsize=(10, 6))
plt.plot(lasso.alphas_, np.sum(lasso.coef_ != 0, axis=0), marker='o')
plt.gca().invert_xaxis()
plt.title("Lasso Regularization Path")
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("Number of Selected Features")
plt.show()



#%%
# =============================================================================
# Method 2: Random Forest Feature Importance
# =============================================================================
# Train a Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Extract feature importances
feature_importances = rf_clf.feature_importances_
important_features_idx = np.argsort(feature_importances)[-20:]  # Select top 20 features

# Select the top features based on importance
X_train_rf = X_train.iloc[:, important_features_idx]
X_test_rf = X_test.iloc[:, important_features_idx]

# Train and evaluate using only important features
clf_rf = LogisticRegression(max_iter=1000, random_state=42)
clf_rf.fit(X_train_rf, y_train)
y_pred_rf = clf_rf.predict(X_test_rf)

balanced_accuracy_rf = balanced_accuracy_score(y_test, y_pred_rf)
print(f"Balanced Accuracy with Random Forest Feature Importance: {balanced_accuracy_rf:.4f}")


#%%
# =============================================================================
# Method 3: Forward Feature Selection - HYBRID 
# =============================================================================
# Use Logistic Regression for forward feature selection
sfs = SFS(LogisticRegression(max_iter=1000, random_state=42),
          k_features=10,  # Select top 10 features
          forward=True,
          floating=False,
          scoring='balanced_accuracy',
          cv=3)

sfs = sfs.fit(X_train, y_train)

# Selected features
selected_features = list(sfs.k_feature_idx_)
X_train_sfs = X_train.iloc[:, selected_features]
X_test_sfs = X_test.iloc[:, selected_features]

# Train and evaluate the model with selected features
clf_sfs = LogisticRegression(max_iter=1000, random_state=42)
clf_sfs.fit(X_train_sfs, y_train)
y_pred_sfs = clf_sfs.predict(X_test_sfs)

balanced_accuracy_sfs = balanced_accuracy_score(y_test, y_pred_sfs)
print(f"Balanced Accuracy with Forward Feature Selection: {balanced_accuracy_sfs:.4f}")

# =============================================================================
# Comparison of Methods
# =============================================================================
print("\nComparison of Balanced Accuracy:")
print(f"PCA: {balanced_accuracy_pca:.4f}")
print(f"Random Forest: {balanced_accuracy_rf:.4f}")
print(f"Forward Feature Selection: {balanced_accuracy_sfs:.4f}")



# %%
# T3 - PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a classifier on the PCA-transformed data
clf_pca = LogisticRegression(max_iter=1000, random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)

# Evaluate the classifier
balanced_accuracy_pca = balanced_accuracy_score(y_test, y_pred_pca)
print(f"Balanced Accuracy with PCA: {balanced_accuracy_pca:.4f}")
