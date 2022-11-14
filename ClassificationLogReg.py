from dataload import *
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import numpy as np


X = np.concatenate((np.ones((X.shape[0],1)),X),1)
Error_train_LRM = 0
Error_test_LRM = 0

chosen_LRM_Comp = 10**(-2)
mu = np.empty((1,M))
sigma = np.empty((1, M))

k = 0


# Normalize data
mu[0, :] = np.mean(X[:, 1:], 0)
sigma[0, :] = np.std(X[:, 1:], 0)

X[:, 1:] = (X[:, 1:] - mu[0, :] ) / sigma[0, :]
X[:, 1:] = (X[:, 1:] - mu[0, :] ) / sigma[0, :]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
LRM_model_i = LogisticRegression(C = 100, max_iter=1000)
LRM_model_i.fit(X_train, y_train)
y_est_LRM_train = LRM_model_i.predict(X_train)
y_est_LRM_test = LRM_model_i.predict(X_test)
Error_train_LRM = np.sum(np.square(y_train - y_est_LRM_train)) / y_train.shape[0]
Error_test_LRM = np.sum(np.square(y_test - y_est_LRM_test)) / y_test.shape[0]

print(LRM_model_i.coef_)

print(Error_test_LRM)
print(chosen_LRM_Comp)