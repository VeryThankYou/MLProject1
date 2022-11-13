from dataload import *
import sklearn.linear_model as lm
from sklearn import model_selection
import numpy as np
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
from main import feature_selector_lr, bmplot

print(attrnames)

# Split dataset into features and target vector
life_idx = attrnames.index('Maximum longevity (yrs)')
y = X[:,life_idx]

X_cols = list(range(0,life_idx)) + list(range(life_idx+1,len(attrnames)))
X = X[:,X_cols]

# Fit ordinary least squares regression model
model = lm.LinearRegression(normalize=True)
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Alcohol content (true)'); ylabel('Alcohol content (estimated)');
subplot(2,1,2)
hist(residual,40)

show()