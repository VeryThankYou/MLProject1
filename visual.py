from dataload import *
import numpy as np
from matplotlib.pyplot import (figure, subplot, boxplot, title, xticks, ylim, show, plot, xlabel, ylabel, yticks, legend)

boxplot(np.log(X))
show()


figure(figsize=(14,7))
for c in range(C):
    subplot(1,C,c+1)
    class_mask = (y==c) # binary mask to extract elements of class c
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c
    
    boxplot(np.log(X[class_mask,:]))
    #title('Class: {0}'.format(classNames[c]))
    title('Class: '+classNames[c])
    xticks(range(1,len(attrnames)+1), [a[:7] for a in attrnames], rotation=45)
    y_up = X.max()+(X.max()-X.min())*0.1; y_down = X.min()-(X.max()-X.min())*0.1
    #ylim(np.log(y_down), np.log(y_up))

show()

# Correlation
figure(figsize=(12,10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attrnames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attrnames[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)

show()