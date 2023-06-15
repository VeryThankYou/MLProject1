import numpy as np
from dataload import *
from sklearn import model_selection

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

errors_out = np.empty((5,1))

for k, (train_index, test_index) in enumerate(CV.split(X,y)):

    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    y_est = np.mean(y_train)
    error = np.square(y_test - y_est).sum() / y_test.shape[0]

    errors_out[k] = error

print(errors_out)