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
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
# ============================================================================
# Question T2.1
# ============================================================================

#%% IMPORT THE DATA
os.chdir('C:/Users/emmaqueen/Documents/ST443/PROJECT/')
current_directory = os.getcwd()
file2 = pd.read_csv('data2.csv.gz')
#print(file2.head())

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
plt.title("Distribution of Active (label=1) vs. Non-Active (label=-1) Compounds")
plt.xlabel("Label")
plt.ylabel("Count")
plt.grid(True)
plt.show()


# ============================================================================
# Question T2.2
# ============================================================================
#%%
# Load the dataset

# removing a lot of features - we have 30 000 features now
file2_new = file2.loc[:, file2.var() > 0.01]

X = file2_new.iloc[:, 1:]  # Features
y = file2_new.iloc[:, 0]   # Target (label)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float32')  # Reduce memory usage
y_train = y_train.astype('float32')

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


#%%
# =============================================================================
# Method 1: Logistic regression with L1 regularization (Lasso)
# =============================================================================



alpha_values = np.logspace(-1, 1, 5)  # Alpha values ranging from 0.0001 to 10000
best_alpha = None
best_balanced_acc_log = 0
mean_cv_score={}



# Loop through alpha values to find the best one using cross-validation
for alpha in alpha_values:
    # Define the Logistic Regression model
    model = LogisticRegression(penalty='l1', solver='saga', C=alpha, max_iter=10000, tol=1e-3, random_state=42,class_weight='balanced')
    
    # Perform cross-validation

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='balanced_accuracy')  # 5-fold cross-validation
    
    # Compute the mean accuracy from cross-validation
    mean_cv_score[alpha] = np.mean(cv_scores)
    
    # Update the best alpha if the current mean accuracy is better
    if mean_cv_score[alpha] > best_balanced_acc_log:
        best_balanced_acc_log = mean_cv_score[alpha]
        best_alpha = alpha

# Print the best alpha and its corresponding cross-validated accuracy
print(f"Best alpha (C): {best_alpha}")
print(f"Best cross-validated balanced accuracy: {best_balanced_acc_log}")

#%%
# see the values of c with the cross vlaidation accuracy
#
# Plot the results
plt.figure(figsize=(8, 6))
plt.semilogx(alpha_values, list(mean_cv_score.values()), marker='o')
plt.xlabel('CAlpha values')
plt.ylabel('Balanced Cross-Validation Accuracy')
plt.title('Balanced Cross-Validation Accuracy for different valeur of Lambda')
plt.grid(True)
plt.show()



#%%
# once we know the parameter alpha we can use it in the logistic function
# Train the final model with the best C
best_logistic_log = LogisticRegression(penalty='l1', solver='saga', C=best_alpha, max_iter=5000, random_state=42)
best_logistic_log.fit(X_train, y_train)

# Make predictions
y_pred_log = best_logistic_log.predict(X_test)

# Evaluate the performance
accuracy_log = accuracy_score(y_test, y_pred_log)
balanced_accuracy_log = balanced_accuracy_score(y_test, y_pred_log)
conf_matrix_log = confusion_matrix(y_test, y_pred_log)
selected_features_log = np.where(best_logistic_log.coef_ != 0)[1]

print(f"Accuracy: {accuracy_log}")
print(f"Balanced Accuracy : {balanced_accuracy_log}")
print(f"Confusion Matrix:\n{conf_matrix_log}")
print(f"Number of selected features: {len(selected_features_log)}")

#%%

#Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_log)
auc = roc_auc_score(y_test, y_pred_log)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve LASSO')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

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
best_k_= 0
best_balanced_acc_tree = 0
best_selected_features_for = None
num_features = []
balanced_accuracies = []


for k in range(1, X_train.shape[1] - 29000):
    selected_features_for = indices[:k]  # Select top k features based on importance
    X_train_k = X_train.iloc[:, selected_features_for]

    # Train Random Forest with k features
    model_k = RandomForestClassifier(random_state=42,n_estimators=20,max_features='log2',class_weight='balanced',max_depth=10)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_k, X_train_k, y_train, cv=cv, scoring='balanced_accuracy') 


    avg_cv_score = np.mean(cv_scores)
    num_features.append(k)
    balanced_accuracies.append(avg_cv_score)

    # Check if the accuracy is better tha the one before
    if avg_cv_score >best_balanced_acc_tree:
        best_k = k 
        best_balanced_acc_tree = avg_cv_score 
        best_selected_features = selected_features_for


print(f"Optimal Number of Features: {best_k}")
print(f"The bast balanced accuracy for the cross validation is : {best_balanced_acc_tree}")



#%% PLOT THE NB OF FEATURES VS BALANCED ACCURACY 

plt.figure(figsize=(8, 6))
plt.plot(num_features, balanced_accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Features')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy vs Number of Features in Random Forest')
plt.grid(True)
plt.show()

#%%

X_train_best = X_train.iloc[:, best_selected_features] 
X_test_best = X_test.iloc[:, best_selected_features]


model_best = RandomForestClassifier(random_state=42, n_estimators=20, max_features='log2', class_weight='balanced', max_depth=10) 
model_best.fit(X_train_best, y_train) 
y_pred_best = model_best.predict(X_test_best)



best_bal_acc_final = balanced_accuracy_score(y_test, y_pred_best) 
conf_matrix_best = confusion_matrix(y_test, y_pred_best)


# Print the results
print("\n--- Best Model Summary ---")
print(f"Best Balanced Accuracy: {best_bal_acc_final:.4f}")
print(f"Best Confusion Matrix:\n{conf_matrix_best}")
print(f"For the final model we have {best_k} features")

#%%
#Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_best)
auc2 = roc_auc_score(y_test, y_pred_best)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {auc2:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve RANDOM FOREST')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#%%
# =============================================================================
# Method 3: Forward Feature Selection - HYBRID 
# =============================================================================


# =============================================================================
# Comparison of the 3 Methods
# =============================================================================






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
plt.grid(True)
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


def soft_threshold(dkj, delta):
    return np.sign(dkj) * np.maximum(0, np.abs(dkj) - delta)

def nearest_shrunken_centroid(X_train, y_train, X_test, delta):
    classes = np.unique(y_train)
    n_features = X_train.shape[1]
    N = len(X_train)

    # Compute overall mean
    overall_mean = X_train.mean(axis=0)

    # Compute centroids for each class
    centroids = np.array([X_train[y_train == cls].mean(axis=0) for cls in classes])

    # Compute pooled variance for each feature
    pooled_var = np.var(X_train, axis=0)
    s0 = np.median(pooled_var)

    # Compute class sizes
    class_sizes = [np.sum(y_train == cls) for cls in classes]

    # Shrink centroids toward the overall mean
    shrunken_centroids = np.zeros_like(centroids)
    selected_features_count = 0  # Counter for selected features

    for i, cls in enumerate(classes):
        for j in range(n_features):
            m_k = class_sizes[i] / N  # Proportional size of class k
            # Compute dkj
            dkj = (centroids[i, j] - overall_mean[j]) / (m_k * (pooled_var[j] + s0))
            # Apply soft thresholding
            dkj_prime = soft_threshold(dkj, delta)
            # Recompute shrunken centroid for feature j in class k
            shrunken_centroids[i, j] = overall_mean[j] + m_k * (pooled_var[j] + s0) * dkj_prime
            if dkj_prime != 0:
                selected_features_count += 1

    # Predict test samples by finding the closest centroid
    y_pred = []
    for x in X_test:
        distances = []
        for i, cls in enumerate(classes):
            dist = np.sum(((x - shrunken_centroids[i]) ** 2) / (pooled_var + s0))  # Distance calculation
            distances.append(dist)
        y_pred.append(classes[np.argmin(distances)])  # Class with minimum distance

    return np.array(y_pred), selected_features_count  # Return predictions and selected feature count

# %% 

# Cross-validation to tune delta
def tune_delta(X, y, deltas, n_splits=5):
    kf = KFold(n_splits=n_splits)
    best_delta = None
    best_accuracy = 0
    best_selected_features_count = 0

    for delta in deltas:
        accuracies = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            y_pred, selected_features_count = nearest_shrunken_centroid(X_train, y_train, X_val, delta)
            accuracies.append(accuracy_score(y_val, y_pred))
        
        mean_accuracy = np.mean(accuracies)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_delta = delta
            best_selected_features_count = selected_features_count  # Track the best number of selected features
    
    return best_delta, best_accuracy, best_selected_features_count


# %% 

# Example usage
deltas = np.linspace(0, 1, 10)  # Test values for delta
best_delta, best_accuracy, best_selected_features_count = tune_delta(X_train, y_train, deltas)
print(f"Best Delta: {best_delta}, Best Cross-Validation Accuracy: {best_accuracy:.4f}")
print(f"Number of Selected Features: {best_selected_features_count}")

# %% 

# Evaluate on the test set with the best delta
y_pred, _ = nearest_shrunken_centroid(X_train, y_train, X_test, delta=best_delta)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# %% 
