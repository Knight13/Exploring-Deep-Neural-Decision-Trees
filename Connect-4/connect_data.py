import numpy as np
import pandas as pd

data = []
with open('connect-4.data', 'r+') as f:
    for line in f:
        line = line.rstrip('\n')
        i = line.split(',')
        data.append(i)

    f.closed

df = pd.DataFrame(data)

df = df.dropna(axis=0, how = "any" )


for i in xrange(df.shape[1]-1):
    name = 'col_'+str(i)
    df = pd.concat([df, pd.get_dummies(df[i], prefix = name)], axis = 1)
    df = df.drop(i, 1)
    
data = df.as_matrix()

labels = []
for i in xrange(data.shape[0]):
    if data[i, 0] == 'win':
        labels.append([1.0, 0.0, 0.0])
    elif data[i,0] == 'loss':
        labels.append([0.0, 1.0, 0.0])
    else:
        labels.append([0.0, 0.0, 1.0])

feature = np.array(data[:, 1:], dtype = np.float32)
label = np.vstack(np.array(i, dtype = np.float32) for i in labels)

