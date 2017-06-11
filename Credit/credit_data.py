import numpy as np
import pandas as pd

df = pd.read_csv("cs-training.csv")

df = df.drop('Unnamed: 0', 1)

df = df.dropna(axis=0, how = "any" )

data = df.as_matrix()

labels = []
for i in xrange(data.shape[0]):
    if data[i, 0] == 1.0:
        labels.append([1.0, 0.0])
    else:
        labels.append([0.0, 1.0])

feature = data[:, 1:]
label = np.vstack(np.array(i, dtype = np.float32) for i in labels)

