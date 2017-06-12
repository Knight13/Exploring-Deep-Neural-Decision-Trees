import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")

df = df.drop('id', 1)

df = df.dropna()

cols = df.columns.tolist()

df_features = df[cols[:-1]]

df_labels = df[cols[-1:]]

df_labels = pd.concat([df_labels, pd.get_dummies(df_labels['target'], prefix = 'target')], axis = 1)

df_labels = df_labels.drop('target', 1)

feature = df_features.as_matrix()
label = df_labels.as_matrix()

