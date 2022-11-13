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
K = 10
CV_out = model_selection.KFold(n_splits=K,shuffle=True)

for train_ix, test_ix in CV_out.split(X):
    # split data 
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
	# configure the cross-validation procedure 
    CV_inner = model_selection.KFold(n_splits=10, shuffle=True)

    for k, (train_index, test_index) in enumerate(CV_inner.split(X_train,y_train)):
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))   
    
        # define the model
        # Extract training and test set for current CV fold, 
        # and convert them to PyTorch tensors
        X_train_2 = torch.Tensor(X_train[train_index,:] )
        y_train_2 = torch.Tensor(y_train[train_index] )
        X_test_2 = torch.Tensor(X_train[test_index,:] )
        y_test_2 = torch.Tensor(y_train[test_index] )
        y_train_2 = y_train.reshape(-1, 1)

        n_hidden_units = 1 # number of hidden units in the signle hidden layer
        # The lambda-syntax defines an anonymous function, which is used here to 
        # make it easy to make new networks within each cross validation fold
        model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(), #torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function
                    )
        # Go to the file 'toolbox_02450.py' in the Tools sub-folder of the toolbox
        # and see how the network is trained (search for 'def train_neural_net',
        # which is the place the function below is defined)

        
        loss_fn = torch.nn.MSELoss()
        # Train for a maximum of 10000 steps, or until convergence (see help for the 
        # function train_neural_net() for more on the tolerance/convergence))
        max_iter = 10000
        print('Training model of type:\n{}\n'.format(str(model())))

        # Do cross-validation:
        errors = [] # make a list for storing generalizaition error in each loop
        # Loop over each cross-validation split. The CV.split-method returns the 
        # indices to be used for training and testing in each split, and calling 
        # the enumerate-method with this simply returns this indices along with 
        # a counter k:

        # Initialize variables
        Features = np.zeros((M,K))
        Error_train = np.empty((K,1))
        Error_test = np.empty((K,1))
        Error_train_fs = np.empty((K,1))
        Error_test_fs = np.empty((K,1))
        Error_train_nofeatures = np.empty((K,1))
        Error_test_nofeatures = np.empty((K,1))

        net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train_2,
                                                        y=y_train_2,
                                                        n_replicates=3,
                                                        max_iter=max_iter)
        
        print('\n\tBest loss: {}\n'.format(final_loss))
        
        # Determine estimated class labels for test set
        y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
        y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
        y_test = y_test.type(dtype=torch.uint8)
        # Determine errors and error rate
        e = (y_test_est != y_test)
        error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
        errors.append(error_rate) # store error rate for current CV fold 












