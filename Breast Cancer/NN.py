import numpy as np
import tensorflow as tf
import cancer_data
import time
import random

x = cancer_data.feature
y = cancer_data.label

input_num_units = x.shape[1]
hidden_num_units_1 = 50
hidden_num_units_2 = 10
num_class = 2

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
hidden_layer_1 = tf.add(tf.matmul(x, weights['hidden_1']), biases['hidden_1'])

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


for i in range(1000):
      _, loss_e = sess.run([train_step, loss], feed_dict={x_ph: x, y_ph: y})
      
      if i % 200 == 0:
          print(loss_e)               
print('error rate %.5f' % (1 - np.mean(np.argmax(y_pred.eval(feed_dict={x_ph: x}), axis=1) == np.argmax(y, axis=1))))

print("--- %s seconds ---" % (time.time() - start_time))



