from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pddata = pd.read_csv("anage.csv")
l = [1, 2, 4, 5, 6, 7, 8, 12, 15, 17, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
l.reverse()
for e in l:
    pddata = pddata.drop(pddata.columns[e], axis=1)


labels = pddata[["HAGRID", "Class"]].to_numpy()
pddata = pddata.drop(pddata.columns[1], axis=1)
X = pddata.to_numpy()
X = X[~np.isnan(X).any(axis = 1)]
idlist = []
for i, e in enumerate(labels):
    exists = False
    for e2 in X:
        if e[0] == int(e2[0]):
            exists = True
            break
    if exists == False:
        idlist.append(i)
labels = np.delete(labels, idlist, 0)

acount = 0
mcount = 0
for e in labels:
    if e[1] == "Aves":
        acount = acount + 1
    else:
        mcount = mcount + 1

print(acount, mcount)

