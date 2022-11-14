from dataload import *
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import *



J = 10
K = 10

CV = KFold(K, shuffle=True)

KNNComp = range(1, 11)
LRMComp = np.logspace(-4, 2, 10)

Error_train_BL = np.empty((K*J,1))
Error_test_BL = np.empty((K*J,1))
Error_train_KNN = np.empty((K*J,1))
Error_test_KNN = np.empty((K*J,1))
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

        # Compute baseline and baseline error
        counter = Counter(y_train)
        BL_Guess = counter.most_common(1)[0][0]
        Error_train_BL[k + i2*J] = np.sum(np.square(y_train - np.ones((y_train.shape)) * BL_Guess)) / y_train.shape[0]
        Error_test_BL[k + i2*J] = np.sum(np.square(y_test - np.ones((y_test.shape)) * BL_Guess)) / y_test.shape[0]

        # Logistic Regression
        int_test_error_LRM = np.empty((internal_cross_validation, 1))
        int_train_error_LRM = np.empty((internal_cross_validation, 1))

        # K-Nearest Neighbors
        int_test_error_KNN = np.empty((internal_cross_validation, 1))
        int_train_error_KNN = np.empty((internal_cross_validation, 1))
        for i in range(internal_cross_validation):
            # Logistic Regression
            LRM_model_i = LogisticRegression(C = 10^2, max_iter=1000)
            LRM_model_i.fit(X_train, y_train)
            y_est_LRM_train = LRM_model_i.predict(X_train)
            y_est_LRM_test = LRM_model_i.predict(X_test)
            int_train_error_LRM[i] = np.sum(np.square(y_train - y_est_LRM_train)) / y_train.shape[0]
            int_test_error_LRM[i] = np.sum(np.square(y_test - y_est_LRM_test)) / y_test.shape[0]

            # K-Nearest Neighbors
            KNN_model_i = KNeighborsClassifier(2)
            KNN_model_i.fit(X_train, y_train)
            y_est_KNN_train = KNN_model_i.predict(X_train)
            y_est_KNN_test = KNN_model_i.predict(X_test)
            int_train_error_KNN[i] = np.sum(np.square(y_train - y_est_KNN_train)) / y_train.shape[0]
            int_test_error_KNN[i] = np.sum(np.square(y_test - y_est_KNN_test)) / y_test.shape[0]
        # Best LRM
        best_model_LRM_id = np.argmin(int_test_error_LRM)
        Error_train_LRM[k + i2*J] = int_train_error_LRM[best_model_LRM_id]
        Error_test_LRM[k + i2*J] = np.min(int_test_error_LRM)

        # Best KNN model
        best_model_KNN_id = np.argmin(int_test_error_KNN)
        Error_train_KNN[k + i2*J] = int_train_error_KNN[best_model_KNN_id]
        Error_test_KNN[k + i2*J] = np.min(int_test_error_KNN)

        nj[k + i2*J] = y_test.shape[0]

        k += 1



rj_LRM_BL = (Error_test_BL - Error_train_LRM) / nj
rj_KNN_BL = (Error_test_BL - Error_train_KNN) / nj
rj_LRM_KNN = (Error_test_KNN - Error_train_LRM) / nj
rho = 1/ K
th_LRM_BL = np.mean(rj_LRM_BL) / (np.std(rj_LRM_BL) * np.sqrt(1/(J*K) + rho / (1 - rho)))
th_KNN_BL = np.mean(rj_KNN_BL) / (np.std(rj_KNN_BL) * np.sqrt(1/(J*K) + rho / (1 - rho)))
th_LRM_KNN = np.mean(rj_LRM_KNN) / (np.std(rj_LRM_KNN) * np.sqrt(1/(J*K) + rho / (1 - rho)))


p_LRM_BL = 2 * t.cdf(- np.abs(th_LRM_BL), df = J*K - 1)
p_KNN_BL = 2 * t.cdf(- np.abs(th_KNN_BL), df = J*K - 1)
p_LRM_KNN = 2 * t.cdf(- np.abs(th_LRM_KNN), df = J*K - 1)

alpha = 0.05
sig_LRM_BL = np.sqrt((1 / 100 + 1 / (K - 1)) * np.var(rj_LRM_BL))
sig_KNN_BL = np.sqrt((1 / 100 + 1 / (K - 1)) * np.var(rj_KNN_BL))
sig_LRM_KNN = np.sqrt((1 / 100 + 1 / (K - 1)) * np.var(rj_LRM_KNN))


conf_int_LRM_BL = [t.ppf(alpha / 2, df = J * K - 1, loc = np.mean(rj_LRM_BL), scale = sig_LRM_BL), t.ppf(1 - alpha / 2, df = J * K - 1, loc = np.mean(rj_LRM_BL), scale = sig_LRM_BL)]
conf_int_KNN_BL = [t.ppf(alpha / 2, df = J * K - 1, loc = np.mean(rj_KNN_BL), scale = sig_KNN_BL), t.ppf(1 - alpha / 2, df = J * K - 1, loc = np.mean(rj_KNN_BL), scale = sig_KNN_BL)]
conf_int_LRM_KNN = [t.ppf(alpha / 2, df = J * K - 1, loc = np.mean(rj_LRM_KNN), scale = sig_LRM_KNN), t.ppf(1 - alpha / 2, df = J * K - 1, loc = np.mean(rj_LRM_KNN), scale = sig_LRM_KNN)]

print(p_LRM_BL)
print(p_KNN_BL)
print(p_LRM_KNN)

print(conf_int_LRM_BL)
print(conf_int_KNN_BL)
print(conf_int_LRM_KNN)