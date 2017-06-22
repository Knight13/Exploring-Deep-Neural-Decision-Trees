import numpy as np
import tensorflow as tf
import otto_data
from neural_network_decision_tree import nn_decision_tree
import time
from sklearn.model_selection import train_test_split
from NNDT_RF import random_forest


x = otto_data.feature
y = otto_data.label

seed = 1990

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=seed)


args = ([X_train, y_train],[X_test, y_test], 10, 100, 10)

start_time = time.time()

pred, test = random_forest(*args)

print('error rate %.5f' % (1 - np.mean(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1))))

print("--- %s seconds ---" % (time.time() - start_time))
