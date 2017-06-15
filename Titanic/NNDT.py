import numpy as np
import tensorflow as tf
import titanic_data
from neural_network_decision_tree import nn_decision_tree
import time
from sklearn.model_selection import train_test_split


x = titanic_data.feature
y = titanic_data.label
d = x.shape[1]

epochs = 100
batch_size = 100

num_cut = []

for features in xrange(d):
    num_cut.append(1)
    
num_leaf = np.prod(np.array(num_cut) + 1)
num_class = y.shape[1]


x_ph = tf.placeholder(tf.float32, [None, d])
y_ph = tf.placeholder(tf.float32, [None, num_class])

cut_points_list = [tf.Variable(tf.random_uniform([i])) for i in num_cut]
leaf_score = tf.Variable(tf.random_uniform([num_leaf, num_class]))

y_pred = nn_decision_tree(x_ph, cut_points_list, leaf_score, temperature=10)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_ph))

seed = 1990

opt = tf.train.AdamOptimizer(0.1)
train_step = opt.minimize(loss)

sess = tf.InteractiveSession()
tf.set_random_seed(1990)

start_time = time.time()

sess.run(tf.initialize_all_variables())

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=seed)

for epoch in range(epochs):
  
  avg_cost = 0
  
  total_batch = int(X_train.shape[0]/batch_size)
  
  for i in range(total_batch):
  
    batch_mask = np.random.choice(X_train.shape[0], batch_size)
    
    
    batch_x = x[batch_mask].reshape(-1, x.shape[1])
    batch_y = y[batch_mask].reshape(-1, y.shape[1])
  
    
    _, loss_e = sess.run([train_step, loss], feed_dict={x_ph: batch_x, y_ph: batch_y})
    
    avg_cost += loss_e / total_batch
    
    
  print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)
  
               
print('error rate %.5f' % (1 - np.mean(np.argmax(y_pred.eval(feed_dict={x_ph: X_test}), axis=1) == np.argmax(y_test, axis=1))))

print("--- %s seconds ---" % (time.time() - start_time))

sess.close()

