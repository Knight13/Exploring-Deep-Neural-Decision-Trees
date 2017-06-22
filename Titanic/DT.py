import numpy as np
import random
import titanic_data
from sklearn import tree
import time
from sklearn.model_selection import train_test_split

x = titanic_data.feature
y = titanic_data.label

seed = random.seed(1990)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=seed)

clf = tree.DecisionTreeClassifier()

start_time = time.time()

clf = clf.fit(X_train, y_train)
      
y_pred = clf.predict(X_test)
        
err.append(1 - np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)))

print('error rate %.5f' %(np.mean(err)))
print("--- %s seconds ---" % (time.time() - start_time))
