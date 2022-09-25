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
print(np.nan)

print(np.where(X != np.nan and (type(X) != int or float)))

print(np.count_nonzero(X, 0))