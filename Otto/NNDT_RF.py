import numpy as np
import tensorflow as tf
import random
from neural_network_decision_tree import nn_decision_tree
from joblib import Parallel, delayed


"""train_data and test_data are list containg the X_train, y_train and X_test, y_test
   obatined after splitting the data set using sklearn.model_selection.train_test_split"""
def random_forest(train_data, test_data, max_features, batch_size, epochs, *args, **kwargs):
  
  #No. of trees defined by diving the total no. of features by the max no. of features in each tree
  #num_trees = int(train_data[0].shape[1]/max_features)
  num_trees = 1

  error = []
  
  for i in xrange(num_trees):
    
    features=[]
    
    for i in xrange(max_features):
      features.append(random.randrange(0,train_data[0].shape[1]))
    
    col_idx = np.array(features)
      
    X_train = train_data[0][:, col_idx]
    y_train = train_data[1]
    
    X_test = test_data[0][:, col_idx]
    y_test = test_data[1]
    
    num_cut = []

    for f in xrange(max_features):
      num_cut.append(1)
      
    num_leaf = np.prod(np.array(num_cut) + 1)
    num_class = y_train.shape[1]

    seed = 1990

    x_ph = tf.placeholder(tf.float32, [None, max_features])
    y_ph = tf.placeholder(tf.float32, [None, num_class])

    cut_points_list = [tf.Variable(tf.random_uniform([i])) for i in num_cut]
    leaf_score = tf.Variable(tf.random_uniform([num_leaf, num_class]))

    y_pred = nn_decision_tree(x_ph, cut_points_list, leaf_score, temperature=10)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_ph))
    
    opt = tf.train.AdamOptimizer(0.1)
    train_step = opt.minimize(loss)
    
    sess = tf.InteractiveSession()
    tf.set_random_seed(1990)

    sess.run(tf.initialize_all_variables())
    
    for epoch in range(epochs):
    
    
       total_batch = int(X_train.shape[0]/batch_size)
    
       for i in range(total_batch):
    
          batch_mask = np.random.choice(X_train.shape[0], batch_size)
      
      
          batch_x = X_train[batch_mask].reshape(-1, X_train.shape[1])
          batch_y = y_train[batch_mask].reshape(-1, y_train.shape[1])
    
      
          _, loss_e = sess.run([train_step, loss], feed_dict={x_ph: batch_x, y_ph: batch_y})
          
    """For each tree, the predicted values and the original y_values are stacked vertically 
       in two different numpy arrays after training each tree for 100 epochs"""
    
    #pred = np.vstack(np.array(y_pred.eval(feed_dict={x_ph: X_test}), dtype = np.float32))
    #orig = np.vstack(np.array(y_test, dtype = np.float32))
    
    return (y_pred.eval(feed_dict={x_ph: X_test}), y_test)
    sess.close()
    
  print('error rate %.5f' % (1 - np.mean(np.argmax(pred, axis=1) == np.argmax(orig, axis=1))))
   
  
     
       

