from dataload import *
from scipy.linalg import svd
import numpy as np
from matplotlib.pyplot import (figure, subplot, boxplot, title, xticks, ylim, show, plot, xlabel, ylabel, yticks, legend, grid, bar)

# Subtract mean value from data
print(np.std(X, 0))
Y = np.divide(X - np.ones((N,1))*X.mean(axis=0), np.std(X, 0))

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9
print(S)
# Plot variance explained
figure()
plot(range(1,len(rho)+1),rho,'x-')
plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plot([1,len(rho)],[threshold, threshold],'k--')
title('Variance explained by principal components')
xlabel('Principal component')
ylabel('Variance explained')
legend(['Individual','Cumulative','Threshold'])
grid()
show()


# Subtract mean value from data
Y = np.divide(X - np.ones((N,1))*X.mean(axis=0), np.ones((N,1))*X.std(axis=0))


# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('AnAge data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()

Y = np.divide(X - np.ones((N,1))*X.mean(axis=0), np.ones((N,1))*X.std(axis=0))

U,S,Vh = svd(Y,full_matrices=False)
V=Vh.T
N,M = X.shape

# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2,3]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    bar(r+i*bw, V[:,i], width=bw)
xticks(r+bw, np.arange(0, 8))
xlabel('Attributes')
ylabel('Component coefficients')
legend(legendStrs)
grid()
title('AnAge: PCA Component Coefficients')
show()