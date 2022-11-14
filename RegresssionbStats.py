from dataload import *
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model as lm
import numpy as np
import torch
from extrascripts import train_neural_net
from scipy.stats import *

# Split dataset into features and target vector
life_idx = attrnames.index('Maximum longevity (yrs)')
y = X[:,life_idx]
attrnames.pop(7)
X_cols = list(range(0,life_idx)) + list(range(life_idx+1,len(attrnames)))
X = X[:,X_cols]

N, M = X.shape

# K-fold CrossValidation

n_hidden_units = 4
max_iter = 10000

loss_fn = torch.nn.MSELoss()

J = 3
K = 10

CV = KFold(K, shuffle=True)

LRMComp = 1 / 1000

Error_train_BL = np.empty((K*J,1))
Error_test_BL = np.empty((K*J,1))
Error_train_ANN = np.empty((K*J,1))
Error_test_ANN = np.empty((K*J,1))
Error_train_LRM = np.empty((K*J,1))
Error_test_LRM = np.empty((K*J,1))
nj = np.empty((K*J,1))

for i2 in range(J):

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

        # Tensors for the ANN
        X_train_ten = torch.from_numpy(X_train).float()
        y_train_ten = torch.from_numpy(y_train).float()
        X_test_ten = torch.from_numpy(X_test).float()
        y_test_ten = torch.from_numpy(y_test).float()
        y_test_ten = torch.reshape(y_test_ten, (-1, 1))
        y_train_ten = torch.reshape(y_train_ten, (-1, 1))
        # Compute baseline and baseline error
        BL_Guess = np.mean(y_train)
        Error_test_BL[k + i2*K] = np.sum(np.square(y_test - np.ones((y_test.shape)) * BL_Guess)) / y_test.shape[0]

        # Linear Regression
        LRM_model_i = lm.Ridge(alpha = LRMComp)
        LRM_model_i.fit(X_train, y_train)
        y_est_LRM_test = LRM_model_i.predict(X_test)
        Error_test_LRM[k + i2*K] = np.sum(np.square(y_test - y_est_LRM_test)) / y_test.shape[0]

        # Neural Network
        model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                        # 1st transfer function, either Tanh or ReLU:
                        torch.nn.Tanh(),                            #torch.nn.ReLU(),
                        torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                        torch.nn.ReLU() # final tranfer function
                        )


        net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train_ten,
                                                        y=y_train_ten,
                                                        n_replicates=1,
                                                        max_iter=max_iter)
        y_est_ANN = net(X_test_ten)

        error = torch.square(y_test_ten - y_est_ANN).sum() / y_test_ten.shape[0]
        Error_test_ANN[k + i2 * K] = int(error)

        nj[k + i2*K] = y_test.shape[0]

        k += 1



rj_LRM_BL = (Error_test_BL - Error_train_LRM) / nj
rj_ANN_BL = (Error_test_BL - Error_train_ANN) / nj
rj_LRM_ANN = (Error_test_ANN - Error_train_LRM) / nj
rho = 1/ K
th_LRM_BL = np.mean(rj_LRM_BL) / (np.std(rj_LRM_BL) * np.sqrt(1/(J*K) + rho / (1 - rho)))
th_ANN_BL = np.mean(rj_ANN_BL) / (np.std(rj_ANN_BL) * np.sqrt(1/(J*K) + rho / (1 - rho)))
th_LRM_ANN = np.mean(rj_LRM_ANN) / (np.std(rj_LRM_ANN) * np.sqrt(1/(J*K) + rho / (1 - rho)))


p_LRM_BL = 2 * t.cdf(- np.abs(th_LRM_BL), df = J*K - 1)
p_ANN_BL = 2 * t.cdf(- np.abs(th_ANN_BL), df = J*K - 1)
p_LRM_ANN = 2 * t.cdf(- np.abs(th_LRM_ANN), df = J*K - 1)

alpha = 0.05
sig_LRM_BL = np.sqrt((1 / (K * J) + 1 / (K - 1)) * np.var(rj_LRM_BL))
sig_ANN_BL = np.sqrt((1 / (K * J) + 1 / (K - 1)) * np.var(rj_ANN_BL))
sig_LRM_ANN = np.sqrt((1 / (K * J) + 1 / (K - 1)) * np.var(rj_LRM_ANN))


conf_int_LRM_BL = [t.ppf(alpha / 2, df = J * K - 1, loc = np.mean(rj_LRM_BL), scale = sig_LRM_BL), t.ppf(1 - alpha / 2, df = J * K - 1, loc = np.mean(rj_LRM_BL), scale = sig_LRM_BL)]
conf_int_ANN_BL = [t.ppf(alpha / 2, df = J * K - 1, loc = np.mean(rj_ANN_BL), scale = sig_ANN_BL), t.ppf(1 - alpha / 2, df = J * K - 1, loc = np.mean(rj_ANN_BL), scale = sig_ANN_BL)]
conf_int_LRM_ANN = [t.ppf(alpha / 2, df = J * K - 1, loc = np.mean(rj_LRM_ANN), scale = sig_LRM_ANN), t.ppf(1 - alpha / 2, df = J * K - 1, loc = np.mean(rj_LRM_ANN), scale = sig_LRM_ANN)]

print(p_LRM_BL)
print(p_ANN_BL)
print(p_LRM_ANN)

print(conf_int_LRM_BL)
print(conf_int_ANN_BL)
print(conf_int_LRM_ANN)


print(Error_test_ANN)
print(Error_test_LRM)