from dataload import *
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

K = 10

CV = KFold(K, shuffle=True)

KNNComp = range(1, 11)
LRMComp = np.logspace(-3, 2, 10)

Error_train_BL = np.empty((K,1))
Error_test_BL = np.empty((K,1))
Error_train_KNN = np.empty((K,1))
Error_test_KNN = np.empty((K,1))
Error_train_LRM = np.empty((K,1))
Error_test_LRM = np.empty((K,1))
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

    # Logistic regression


    k += 1
print(Error_train_BL)
print(Error_test_BL)
