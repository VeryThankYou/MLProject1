import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
from extrascripts import feature_selector_lr, train_neural_net, draw_neural_net, visualize_decision_boundary
import torch
from dataload import *
import sklearn.linear_model as lm

# Split dataset into features and target vector
life_idx = attrnames.index('Maximum longevity (yrs)')
y = X[:,life_idx]
attrnames.pop(7)
X_cols = list(range(0,life_idx)) + list(range(life_idx+1,len(attrnames)))
X = X[:,X_cols]

N, M = X.shape

# K-fold CrossValidation
K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)
reg_parameter = np.logspace(-3, 2, 5)

mu = np.empty((K, M))
sigma = np.empty((K, M))
# Do cross-validation:
errors_out = np.empty((5,1))

# Chosen regularization parameter
CRP = np.empty((5,1))

max_iter = 10000

for k, (train_index, test_index) in enumerate(CV.split(X,y)):

    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Normalize data
    mu[k, :] = np.mean(X_train, 0)
    sigma[k, :] = np.std(X_train, 0)
    
    X_train = (X_train - mu[k, :] ) / sigma[k, :]
    X_test = (X_test - mu[k, :] ) / sigma[k, :]
    
    # Internal error:
    int_error = np.empty((K,1))

    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    for i in range(K): 

        linear_model = lm.Ridge(alpha=reg_parameter[i])

        linear_model.fit(X_train, y_train)

        y_est = linear_model.predict(X_test)
        error = np.square(y_test - y_est).sum() / y_test.shape[0]
        print(error)

        int_error[i] = int(error)
        
    
    id = np.argmin(int_error)
    errors_out[k] = min(int_error)
    CRP[k] = reg_parameter[id]

print(errors_out, CRP)