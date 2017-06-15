import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")


df = pd.concat([df, pd.get_dummies(df['Sex'], prefix = 'Sex')], axis = 1)
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix = 'Embarked')], axis = 1)


df = df.drop(['PassengerId','Name', 'Ticket', 'Cabin', 'Sex', 'Embarked'], 1)

df = df.dropna(axis=0, how = "any" )

data = df.as_matrix()


labels = []
for i in xrange(data.shape[0]):
    if data[i, 0] == 1.0:
        labels.append([1.0, 0.0])
    else:
        labels.append([0.0, 1.0])
        

feature = np.array(data[:, 1:], dtype = np.float32)
label = np.vstack(np.array(i, dtype = np.float32) for i in labels)

