import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pddata = pd.read_csv("anage.csv")
raw_data = pddata.values
cols = len(pddata.columns)
attributeNames = np.asarray(pddata.columns)
X = raw_data[:, cols - 1]
print(attributeNames)
