import numpy as np
import pandas as pd

pddata = pd.read_csv("anage.csv")
l = [1, 2, 4, 5, 6, 7, 8, 12, 15, 17, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
l.reverse()
for e in l:
    pddata = pddata.drop(pddata.columns[e], axis=1)



labels = pddata[["HAGRID", "Class"]].to_numpy()
pddata = pddata.drop(pddata.columns[1], axis=1)
attrnames = pddata.columns
attrnames = list(attrnames.delete(0))
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
X = np.delete(X, 0, 1)
labels = np.delete(labels, idlist, 0)
labels = np.reshape(np.delete(labels, 0, 1), (1, -1))[0]
classNames = list(set(labels))
classDict = dict(zip(classNames, range(2)))
y = np.asarray([classDict[value] for value in labels])
N = len(y)
M = len(attrnames)
C = len(classNames)

print(np.std(X, 0))
#print(np.sum(y))
#print(classDict)

