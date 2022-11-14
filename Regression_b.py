import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
from extrascripts import feature_selector_lr, train_neural_net, draw_neural_net, visualize_decision_boundary
import torch
from dataload import *
import sklearn.linear_model as lm
plt.rcParams.update({'font.size': 12})


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
n_hidden_units = range(1,6)

mu = np.empty((K, M))
sigma = np.empty((K, M))
# Do cross-validation:
errors_out = np.empty((5,1))

# Chosen hidden unit
CHU = np.empty((5,1))

loss_fn = torch.nn.MSELoss()
        # Train for a maximum of 10000 steps, or until convergence (see help for the 
        # function train_neural_net() for more on the tolerance/convergence))

max_iter = 10000


for k, (train_index, test_index) in enumerate(CV.split(X,y)):

    X_train = X[train_index,:]
    X_test = X[test_index,:]

    # Normalize data
    mu[k, :] = np.mean(X_train, 0)
    sigma[k, :] = np.std(X_train, 0)
    
    X_train = (X_train - mu[k, :] ) / sigma[k, :]
    X_test = (X_test - mu[k, :] ) / sigma[k, :]


    # Extract training and test set for current CV fold, 
    # and convert them to PyTorch tensors
    X_train = torch.Tensor(X[train_index,:] )
    y_train = torch.Tensor(y[train_index] )
    X_test = torch.Tensor(X[test_index,:] )
    y_test = torch.Tensor(y[test_index] )
    y_train = y_train.reshape(-1, 1)

    

    # Internal error:
    int_error = np.empty((K,1))

    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    for i in range(K): 

        
        model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden_units[i]), #M features to H hiden units
                            # 1st transfer function, either Tanh or ReLU:
                            torch.nn.Tanh(),                            #torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_units[i], 1), # H hidden units to 1 output neuron
                            torch.nn.ReLU() # final tranfer function
                            )

        print('Training model of type:\n{}\n'.format(str(model())))

        net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train,
                                                        y=y_train,
                                                        n_replicates=3,
                                                        max_iter=max_iter)
        y_est = net(X_test)
        print(y_est)
        error = torch.square(y_test - y_est).sum() / y_test.shape[0]
        print(error)


        int_error[i] = int(error)
        
    
    id = np.argmin(int_error)
    errors_out[k] = min(int_error)
    CHU[k] = n_hidden_units[id]

print(errors_out, CHU)