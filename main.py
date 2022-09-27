import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pddata = pd.read_csv("anage.csv")
raw_data = pddata.values
cols = len(pddata.columns)
attributeNames = np.asarray(pddata.columns)


X = np.asarray(raw_data)
print(X[1])
X = np.delete(X, [0, 1, 2, 3, 4, 5, 6, 7, 8, 22, 23, 24, 25, 26, 27, 29, 30], 1)

print(X[1])

print(pddata.isnull().sum())
print(type(X[0, 0]))
print(X[0, 0])
np.count_nonzero(np.isnan(X))
count = (X == np.nan).sum()
print(count)