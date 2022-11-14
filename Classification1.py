from dataload import *
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

K = 10

CV = KFold(K, shuffle=True)

KNNComp = range(1, 11)
LRMComp = np.logspace(-4, 2, 10)

Error_train_BL = np.empty((K,1))
Error_test_BL = np.empty((K,1))
Error_train_KNN = np.empty((K,1))
Error_test_KNN = np.empty((K,1))
Error_train_LRM = np.empty((K,1))
Error_test_LRM = np.empty((K,1))
chosen_KNN_Comp = np.empty((K, 1))
chosen_LRM_Comp = np.empty((K, 1))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))

k = 0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10

    # Normalize data
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :]

    # Compute baseline and baseline error
    counter = Counter(y_train)
    BL_Guess = counter.most_common(1)[0][0]
    Error_train_BL[k] = np.sum(np.square(y_train - np.ones((y_train.shape)) * BL_Guess)) / y_train.shape[0]
    Error_test_BL[k] = np.sum(np.square(y_test - np.ones((y_test.shape)) * BL_Guess)) / y_test.shape[0]

    # Logistic Regression
    int_test_error_LRM = np.empty((internal_cross_validation, 1))
    int_train_error_LRM = np.empty((internal_cross_validation, 1))

    # K-Nearest Neighbors
    int_test_error_KNN = np.empty((internal_cross_validation, 1))
    int_train_error_KNN = np.empty((internal_cross_validation, 1))
    for i in range(internal_cross_validation):
        # Logistic Regression
        LRM_model_i = LogisticRegression(C = 1/LRMComp[i], max_iter=1000)
        LRM_model_i.fit(X_train, y_train)
        y_est_LRM_train = LRM_model_i.predict(X_train)
        y_est_LRM_test = LRM_model_i.predict(X_test)
        int_train_error_LRM[i] = np.sum(np.square(y_train - y_est_LRM_train)) / y_train.shape[0]
        int_test_error_LRM[i] = np.sum(np.square(y_test - y_est_LRM_test)) / y_test.shape[0]

        # K-Nearest Neighbors
        KNN_model_i = KNeighborsClassifier(KNNComp[i])
        KNN_model_i.fit(X_train, y_train)
        y_est_KNN_train = KNN_model_i.predict(X_train)
        y_est_KNN_test = KNN_model_i.predict(X_test)
        int_train_error_KNN[i] = np.sum(np.square(y_train - y_est_KNN_train)) / y_train.shape[0]
        int_test_error_KNN[i] = np.sum(np.square(y_test - y_est_KNN_test)) / y_test.shape[0]
    # Best LRM
    best_model_LRM_id = np.argmin(int_test_error_LRM)
    chosen_LRM_Comp[k] = LRMComp[best_model_LRM_id]
    Error_train_LRM[k] = int_train_error_LRM[best_model_LRM_id]
    Error_test_LRM[k] = np.min(int_test_error_LRM)

    # Best KNN model
    best_model_KNN_id = np.argmin(int_test_error_KNN)
    chosen_KNN_Comp[k] = KNNComp[best_model_KNN_id]
    Error_train_KNN[k] = int_train_error_KNN[best_model_KNN_id]
    Error_test_KNN[k] = np.min(int_test_error_KNN)


    k += 1

print(Error_test_BL)
print(Error_test_KNN)
print(Error_test_LRM)
print(chosen_KNN_Comp)
print(chosen_LRM_Comp)