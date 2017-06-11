import numpy as np
import tensorflow as tf
import credit_data
from neural_network_decision_tree import nn_decision_tree
import time



x = credit_data.feature
y = credit_data.label
d = x.shape[1]

num_cut = []

for features in xrange(d):
    num_cut.append(1)
    
num_leaf = np.prod(np.array(num_cut) + 1)
num_class = 2


x_ph = tf.placeholder(tf.float32, [None, d])
y_ph = tf.placeholder(tf.float32, [None, num_class])

cut_points_list = [tf.Variable(tf.random_uniform([i])) for i in num_cut]
leaf_score = tf.Variable(tf.random_uniform([num_leaf, num_class])

y_pred = nn_decision_tree(x_ph, cut_points_list, leaf_score, temperature=10)

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
