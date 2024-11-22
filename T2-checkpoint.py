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
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.neighbors import NearestCentroid


# ============================================================================
# Question T2.1
# ============================================================================

#%% IMPORT THE DATA
os.chdir('C:/Users/emmaqueen/Documents/ST443/PROJECT/')
current_directory = os.getcwd()
file2 = pd.read_csv('data2.csv.gz')
print(file2.head())

#%%
#SUMMARY OF OUR DATA

shape = file2.shape
info = file2.info()
describe = file2.describe
counts_of_actives_cells = file2["label"].value_counts()
missing_values= file2.isnull().sum().sum()
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
plt.grid(True)
plt.show()




# ============================================================================
# Question T2.2
# ============================================================================
#%%
# Load the dataset

X = file2.iloc[:, 1:]  # Features
y = file2.iloc[:, 0]   # Target (label)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float32')  # Reduce memory usage
y_train = y_train.astype('float32')

#%%
# =============================================================================
# Method 1: Logistic regression with L1 regularization (Lasso)
# =============================================================================

# Logistic Regression with L1 regularization
logistic = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=42)# Make predictions

# Define a grid of potential C values
param_grid = {'C': np.logspace(-4, 4, 50)}  # Search range for C
grid_search = GridSearchCV(estimator=logistic, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameter
best_c = grid_search.best_params_['C']
print(f"Optimal C: {best_c}")


#%%
# see the values of c with the cross vlaidation accuracy
# Extract results from GridSearchCV
results = grid_search.cv_results_
c_values = param_grid['C']
mean_accuracies = results['mean_test_score']

# Plot the results
plt.figure(figsize=(8, 6))
plt.semilogx(c_values, mean_accuracies, marker='o')
plt.xlabel('C (Inverse of Regularization Strength)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Cross-Validation Accuracy vs. C')
plt.grid(True)
plt.show()



#%%
# once we know the parameter alpha we can use it in the logistic function
# Train the final model with the best C
best_logistic = LogisticRegression(penalty='l1', solver='saga', C=best_c, max_iter=5000, random_state=42)
best_logistic.fit(X_train, y_train)

# Make predictions
y_pred = best_logistic.predict(X_test)

# Evaluate the performance
accuracy_log = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy_log}")
print(f"Confusion Matrix:\n{conf_matrix}")


#%%
# Extract non-zero coefficients (selected features)
selected_features_log = np.where(best_logistic.coef_ != 0)[1]
print(f"Number of selected features: {len(selected_features_log)}")


#%%
# =============================================================================
# Method 2: Random Forest Feature Importance
# =============================================================================
# Train a Random Forest classifier on all the data
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
#%%
# Rank the features importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort in descending order

#%%
# Loop to find a the best number of k features
best_k = 0
best_balanced_acc = 0
best_selected_features_for = None
best_conf_matrix = None  

for k in range(1, X_train.shape[1] + 1):
    selected_features_for = indices[:k]  # Select top k features based on importance
    X_train_k = X_train.iloc[:, selected_features_for]
    X_test_k = X_test.iloc[:, selected_features_for]

    # Train Random Forest with k features
    model_k = RandomForestClassifier(random_state=42)
    model_k.fit(X_train_k, y_train)
    y_pred_k = model_k.predict(X_test_k)

    
    balanced_acc_k = balanced_accuracy_score(y_test, y_pred_k)
    conf_matrix_k = confusion_matrix(y_test, y_pred_k)

    # Check if the accuracy is better tha the one before
    if balanced_acc_k > best_balanced_acc:
        best_k = k
        best_balanced_acc = balanced_acc_k
        best_selected_features_for = selected_features_for
        best_conf_matrix = conf_matrix_k  # Store confusion matrix for the best model

# Print the results
print("\n--- Best Model Summary ---")
print(f"Optimal Number of Features: {best_k}")
print(f"Best Balanced Accuracy: {best_balanced_acc:.4f}")
print(f"Best Confusion Matrix:\n{best_conf_matrix}")

#%%
# =============================================================================
# Method 3: Forward Feature Selection - HYBRID 
# =============================================================================


# =============================================================================
# Comparison of the 3 Methods
# =============================================================================


best_models = {
    'logistic with lasso penalization': {
        'model': best_logistic,
        'features': selected_features_log
    },
    'random_forest': {
        'model': best_selected_features_for,
        'features': best_selected_features_for
    }
}




#%%
# ============================================================================
# Question T2.3
# ============================================================================

# %%
# ============================================================================
# Elastic Net logistic Regression - mix between Lasso and Ridge
# ============================================================================

elastic_net = ElasticNet(alpha = 0.1, l1_ratio= 0.5, max_iter= 5000, random_state= 42)
elastic_net.fit(X_train,y_train)


param_grid2 = {
    'alpha' = [0.1, 1, 10],
    'l1_ratio' = [0.2, 0.5, 0.8]
}

grid_search2 = GridSearchCV(estimator=elastic_net, param_grid=param_grid2)
grid_search.fit(X_train,y_train)

best_alpha1 = grid_search2.best_estimator_.alpha
best_l1ratio = grid_search2.best_estimator_.l1_ratio

print('Best alpha : ', best_alpha1)
print('Best l1 ratio : ', best_l1ratio)

#%%

results2 = grid_search2.cv_results_
alpha_values = param_grid2['alpha']
l1_balues = param_grid2['l1_ratio']
mean_accuracies2 = results2['mean_test_score']


# Plot the results
plt.figure(figsize=(8, 6))
plt.semilogx(alpha_values, mean_accuracies2, marker='o')
plt.semilogx(l1_balues, mean_accuracies2, marker='x' )
plt.grid(True)
plt.show()


#%%

elastic_net_best = LogisticRegression(alpha = best_alpha1, l1_ratio= best_l1ratio, max_iter= 5000, random_state= 42)
elastic_net_best.fit(X_train,y_train)

y_red_elastic = elastic_net_best.predict(X_test)

accuracy_elastic = accuracy_score(y_test,y_red_elastic)
confusion_matrix_elastic = confusion_matrix(y_test,y_red_elastic)

print(f"Accuracy :{accuracy_elastic}")
print(f"Confusion Matrix : \n{confusion_matrix_elastic}")

#%%
selected_features_lelastic = np.where(elastic_net_best.coef_ != 0)[1]
print(f"Number of selected features: {len(selected_features_lelastic)}")


# %%
# ============================================================================
# Method nearest Shrunken Centroid 
# ============================================================================

#%%

