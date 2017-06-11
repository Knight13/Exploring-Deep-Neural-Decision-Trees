import numpy as np
import pandas as pd

df = pd.read_csv("Watch_accelerometer.csv")

df = df.dropna()

df = df[df['gt'] != 'null']

cols = df.columns.tolist()

df_features = df[cols[:-1]]

df_features = pd.concat([df_features, pd.get_dummies(df_features['User'], prefix = 'User')], axis = 1)
df_features = pd.concat([df_features, pd.get_dummies(df_features['Model'], prefix = 'Model')], axis = 1)
df_features = pd.concat([df_features, pd.get_dummies(df_features['Device'], prefix = 'Device')], axis = 1)

df_features = df_features.drop(['Index', 'User', 'Model', 'Device'], 1)

df_labels = df[cols[-1:]]

df_labels = pd.concat([df_labels, pd.get_dummies(df_labels['gt'], prefix = 'gt')], axis = 1)

df_labels = df_labels.drop('gt', 1)



feature = df_features.as_matrix()
label = df_labels.as_matrix()

