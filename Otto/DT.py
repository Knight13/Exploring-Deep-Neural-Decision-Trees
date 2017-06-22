import numpy as np
import random
import otto_data
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

x = otto_data.feature
y = otto_data.label

seed = random.seed(1990)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=seed)

features = 10

num_trees = int(X_train.shape[1]/features)


clf = RandomForestClassifier(n_estimators=num_trees, max_features=features)

start_time = time.time()

clf = clf.fit(X_train, y_train)
      
y_pred = clf.predict(X_test)
        
np.mean(1 - np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)))

print('error rate %.5f' %(np.mean(1 - np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)))))
print("--- %s seconds ---" % (time.time() - start_time))
