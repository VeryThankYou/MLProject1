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

outer_results = list()

for train_ix, test_ix in CV_out.split(X):
    # split data 
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
	# configure the cross-validation procedure 
    CV_inner = model_selection.KFold(n_splits=10, shuffle=True)

    for k, (train_index, test_index) in enumerate(CV_inner.split(X,y)):
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))   
    
        # define the model
        # Extract training and test set for current CV fold, 
        # and convert them to PyTorch tensors
        X_train = torch.Tensor(X[train_index,:] )
        y_train = torch.Tensor(y[train_index] )
        X_test = torch.Tensor(X[test_index,:] )
        y_test = torch.Tensor(y[test_index] )
        y_train = y_train.reshape(-1, 1)
        # Go to the file 'toolbox_02450.py' in the Tools sub-folder of the toolbox
        # and see how the network is trained (search for 'def train_neural_net',
        # which is the place the function below is defined)
        net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train,
                                                        y=y_train,
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










# Define the model structure
n_hidden_units = 1 # number of hidden units in the signle hidden layer
# The lambda-syntax defines an anonymous function, which is used here to 
# make it easy to make new networks within each cross validation fold
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(),                            #torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function
                    )
# Since we're training a neural network for binary classification, we use a 
# binary cross entropy loss (see the help(train_neural_net) for more on
# the loss_fn input to the function)
loss_fn = torch.nn.BCELoss()
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



 
    
    
    

    
    
#     # visualize_decision_boundary(predict, X, y, # provide data, along with function for prediction
#     #                             attrnames, classNames, # provide information on attribute and class names
#     #                             train=train_index, test=test_index, # provide information on partioning
#     #                             show_legend=k==(K-1)) # only display legend for last plot
    
    
#     # Display the learning curve for the best net in the current fold
#     h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
#     h.set_label('CV fold {0}'.format(k+1))
#     summaries_axes[0].set_xlabel('Iterations')
#     summaries_axes[0].set_xlim((0, max_iter))
#     summaries_axes[0].set_ylabel('Loss')
#     summaries_axes[0].set_title('Learning curves')
    
# # Display the error rate across folds
# summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
# summaries_axes[1].set_xlabel('Fold')
# summaries_axes[1].set_xticks(np.arange(1, K+1))
# summaries_axes[1].set_ylabel('Error rate')
# summaries_axes[1].set_title('Test misclassification rates')
    
# # Show the plots
# # plt.show(decision_boundaries.number) # try these lines if the following code fails (depends on package versions)
# # plt.show(summaries.number)
# plt.show()

# # Display a diagram of the best network in last fold
# print('Diagram of best neural net in last fold:')
# weights = [net[i].weight.data.numpy().T for i in [0,2]]
# biases = [net[i].bias.data.numpy() for i in [0,2]]
# tf =  [str(net[i]) for i in [1,3]]
# draw_neural_net(weights, biases, tf)

# # Print the average classification error rate
# print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))

# print('Ran exercise 8.2.2.')