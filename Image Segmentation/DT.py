import numpy as np
import random
import image_data
import tensorflow as tf
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import time


x = image_data.feature
y = image_data.label

seed = random.seed(1990)
kf = KFold(n_splits=100, random_state=seed, shuffle= True)

clf = tree.DecisionTreeClassifier()

loss = 0
iteration = 0
start_time = time.time()

for i in range(1000):
  for train_index, test_index in kf.split(x):
      
      X_train, X_test = x[train_index], x[test_index]
      y_train, y_test = y[train_index], y[test_index]
      clf = clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      loss += log_loss(y_test, y_pred)
      iteration += 1

print('error rate %.2f' %(loss/iteration))
print("--- %s seconds ---" % (time.time() - start_time))
