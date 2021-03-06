import numpy as np
import tensorflow as tf
import flight_data
import time
import random
from sklearn.model_selection import train_test_split
   
x = flight_data.feature
y = flight_data.label

epochs = 100
batch_size = 100

input_num_units = x.shape[1]
hidden_num_units_1 = 210
hidden_num_units_2 = 300
num_class = y.shape[1]

sess = tf.InteractiveSession()
tf.set_random_seed(1990)


x_ph = tf.placeholder(tf.float32, [None, input_num_units])
y_ph = tf.placeholder(tf.float32, [None, num_class])

seed = 1990

weights = {
    'hidden_1': tf.Variable(tf.random_normal([input_num_units, hidden_num_units_1], seed=seed)),
    'hidden_2': tf.Variable(tf.random_normal([hidden_num_units_1, hidden_num_units_2], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units_2, num_class], seed=seed))
}

biases = {
    'hidden_1': tf.Variable(tf.random_normal([hidden_num_units_1], seed=seed)),
    'hidden_2': tf.Variable(tf.random_normal([hidden_num_units_2], seed=seed)),
    'output': tf.Variable(tf.random_normal([num_class], seed=seed))
}

x = np.array(x, dtype = np.float32)

#1st hidden layer
hidden_layer_1 = tf.add(tf.matmul(x_ph, weights['hidden_1']), biases['hidden_1'])

hidden_layer_1 = tf.nn.softmax(hidden_layer_1)

#2nd hidden layer
hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weights['hidden_2']), biases['hidden_2'])

hidden_layer_2 = tf.nn.softmax(hidden_layer_2)

#output layer
y_pred = tf.matmul(hidden_layer_2, weights['output']) + biases['output']

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_ph))


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
  
batch_mask = np.random.choice(X_test.shape[0], X_test.shape[0]/10)
               
print('error rate %.5f' % (1 - np.mean(np.argmax(y_pred.eval(feed_dict={x_ph: X_test[batch_mask].reshape(-1, X_test.shape[1])}), axis=1) == np.argmax(y_test[batch_mask].reshape(-1, y_test.shape[1]), axis=1))))


print("--- %s seconds ---" % (time.time() - start_time))

sess.close()




