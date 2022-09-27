import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pddata = pd.read_csv("anage.csv")
l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 17, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
l.reverse()
for e in l:
    pddata = pddata.drop(pddata.columns[e], axis=1)
print(pddata)
print(pddata.isnull().sum())



X = pddata.to_numpy()


# print(pddata.isnull().sum())
# print(type(X[0, 0]))
# print(X[0, 0])
X = X[~np.isnan(X).any(axis = 1)]
print(X.shape)