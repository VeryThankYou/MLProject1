import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pddata = pd.read_csv("anage.csv")
raw_data = pddata.values
cols = len(pddata.columns)
attributeNames = np.asarray(pddata.columns)


X = np.asarray(raw_data)

X = np.delete(X, range(0, 10), 1)
X = np.delete(X, [12, 13, 14, -1], 1)
print(X)
print(pddata.isnull().sum())
print(type(X[0, 0]))
print(X[0, 0])

count = (X == np.nan).sum()
print(count)