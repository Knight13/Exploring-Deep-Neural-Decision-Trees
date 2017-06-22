import numpy as np
import random
import image_data
from sklearn import tree
import time
from sklearn.model_selection import train_test_split

x = image_data.feature
y = image_data.label

seed = random.seed(1990)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=seed)

clf = tree.DecisionTreeClassifier()

start_time = time.time()

clf = clf.fit(X_train, y_train)
      
y_pred = clf.predict(X_test)

print('error rate %.5f' %(np.mean(1 - np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)))))
print("--- %s seconds ---" % (time.time() - start_time))
