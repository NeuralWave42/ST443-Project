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
from sklearn.model_selection import cross_val_score

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
#print(" The statistic summary is : ", describe)
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


file2_new = file2.loc[:, file2.var() > 0.01]

X = file2_new.iloc[:, 1:]  # Features
y = file2_new.iloc[:, 0]   # Target (label)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float32')  # Reduce memory usage
y_train = y_train.astype('float32')




#%%
# =============================================================================
# Method 1: Logistic regression with L1 regularization (Lasso)
# =============================================================================


alpha_values = np.logspace(-4, 1, 15)  # Alpha values ranging from 0.0001 to 10000
best_alpha = None
best_cv_score = 0
mean_cv_score={}

# Loop through alpha values to find the best one using cross-validation
for alpha in alpha_values:
    # Define the Logistic Regression model
    model = LogisticRegression(penalty='l1', solver='saga', C=alpha, max_iter=10000, tol=1e-3, random_state=42,class_weight='balanced')
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
    
    # Compute the mean accuracy from cross-validation
    mean_cv_score[alpha] = np.mean(cv_scores)
    
    # Update the best alpha if the current mean accuracy is better
    if mean_cv_score[alpha] > best_cv_score:
        best_cv_score = mean_cv_score[alpha]
        best_alpha = alpha

# Print the best alpha and its corresponding cross-validated accuracy
print(f"Best alpha (C): {best_alpha}")
print(f"Best cross-validated accuracy: {best_cv_score}")

#%%
# see the values of c with the cross vlaidation accuracy
#
# Plot the results
plt.figure(figsize=(8, 6))
plt.semilogx(alpha_values, list(mean_cv_score.values()), marker='o')
plt.xlabel('C (Inverse of Regularization Strength)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Cross-Validation Accuracy vs. C')
plt.grid(True)
plt.show()



#%%
# once we know the parameter alpha we can use it in the logistic function
# Train the final model with the best C
best_logistic = LogisticRegression(penalty='l1', solver='saga', C=best_alpha, max_iter=5000, random_state=42)
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
model = RandomForestClassifier(random_state=42,)
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

for k in range(1, X_train.shape[1] - 29000):
    selected_features_for = indices[:k]  # Select top k features based on importance
    X_train_k = X_train.iloc[:, selected_features_for]
    X_test_k = X_test.iloc[:, selected_features_for]

    # Train Random Forest with k features
    model_k = RandomForestClassifier(random_state=42,n_estimators=20,max_features='log2',class_weight='balanced',max_depth=10)
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


# Define the model with ElasticNet regularization
model = LogisticRegression(
    penalty='elasticnet',  # ElasticNet regularization
    solver='saga',         # Solver that supports ElasticNet  # Multinomial logistic regression
    l1_ratio=0.5,          # α (balance between L1 and L2 regularization)
    C=1.0,
    max_iter=10000,
    tol=1e-3,
    class_weight='balanced'                # Regularization strength (inverse of λ)
)

model.fit(X_train,y_train)

param_grid2 = {
    'C': [0.01, 0.05, 0.1],  # Fewer choices for C
    'l1_ratio': [0.3, 0.5, 0.7]  # Fewer choices for l1_ratio
}


grid_search2 = GridSearchCV(estimator=model, param_grid=param_grid2)
grid_search2.fit(X_train,y_train)

best_C1 = grid_search2.best_estimator_.C
best_l1ratio = grid_search2.best_estimator_.l1_ratio

print('Best alpha : ', best_C1)
print('Best l1 ratio : ', best_l1ratio)

#%%

results2 = grid_search2.cv_results_
alpha_values = param_grid2['C']
l1_values = param_grid2['l1_ratio']
mean_accuracies2 = results2['mean_test_score']

# Reshape mean accuracies to match the grid shape (len(l1_values) x len(alpha_values))
mean_accuracies2 = mean_accuracies2.reshape(len(l1_values), len(alpha_values))

# Plot the results
plt.figure(figsize=(10, 8))
for i, l1 in enumerate(l1_values):
    plt.semilogx(alpha_values, mean_accuracies2[i, :], marker='o', label=f'l1_ratio={l1}')

plt.xlabel("C (Inverse Regularization Strength)")
plt.ylabel("Mean Cross-Validated Accuracy")
plt.title("Grid Search Results for ElasticNet Logistic Regression")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

#%%

model = LogisticRegression(
    penalty='elasticnet',  # ElasticNet regularization
    solver='saga',         # Solver that supports ElasticNet  # Multinomial logistic regression
    l1_ratio=0.7,          # α (balance between L1 and L2 regularization)
    C=0.05,
    max_iter=1000,
    tol=1e-3,
    class_weight='balanced'                # Regularization strength (inverse of λ)
)

model.fit(X_train,y_train)
y_red_elastic = model.predict(X_test)

accuracy_elastic = accuracy_score(y_test,y_red_elastic)
confusion_matrix_elastic = confusion_matrix(y_test,y_red_elastic)

print(f"Accuracy :{accuracy_elastic}")
print(f"Confusion Matrix : \n{confusion_matrix_elastic}")

#%%
selected_features_lelastic = np.where(model.coef_ != 0)[1]
print(f"Number of selected features: {len(selected_features_lelastic)}")


# %%
# ============================================================================
# Method nearest Shrunken Centroid 
# ============================================================================



