#!/disk/scratch/mlp/miniconda2/bin/python

from __future__ import division
import numpy as np
import tensorflow as tf
from functools import reduce
import time
import random
from math import ceil

'''This class assumes equal number of cuts per feature'''

class NNDT:

    def __init__(self, train_data, test_data, cut_per_feature = 1, beta = 0.01,
                temperature = 100, regularizer = False, batch_size = 100, epochs = 100, *args, **kwargs):

        self.epochs = epochs
        self.batch_size = batch_size

        self.X_train = train_data[0]
        self.y_train = train_data[1]

        self.X_test = test_data[0]
        self.y_test = test_data[1]

        self.temperature = temperature

        self.beta = beta
        self.regularizer = regularizer

        self.time = 0

        self.train_error = 0
        self.test_error = 0

        self.cuts = cut_per_feature

        self.W = tf.reshape(tf.linspace(1.0, self.cuts + 1.0, self.cuts + 1), [1, -1])

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def tf_kron_prod(self, a, b):
        dim_a = a.get_shape().as_list()[-1]
        dim_b = b.get_shape().as_list()[-1]
        res = tf.reshape(tf.matmul(tf.reshape(a, [-1, dim_a, 1]), tf.reshape(b, [-1, 1, dim_b])), [-1, dim_a * dim_b])
        return res

    def tf_bin(self, x, cut_points):
        # x is a N-by-1 matrix (column vector)
        # cut_points is a D-dim vector (D is the number of cut-points)
        # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
        x = tf.cast(x, tf.float32)
        cut_points = -tf.nn.top_k(-cut_points, self.cuts)[0]  # make sure cut_points is monotonically increasing
        b = tf.cumsum(tf.concat([tf.constant(0.0, shape=[1]), -cut_points],0))
        h = tf.matmul(x, self.W) + b
        res = tf.nn.softmax(h * self.temperature)
        return res


    def nn_decision_tree(self, x, cut_points_list, leaf_score):
        # cut_points_list contains the cut_points for each dimension of feature
        return tf.sparse_matmul(reduce(self.tf_kron_prod, map(lambda z: self.tf_bin(x[:, z[0]:z[0] + 1], z[1]), enumerate(cut_points_list))), leaf_score, a_is_sparse=True, b_is_sparse=True)

    def fit_tree(self):

        num_bin = []

        cut_points_list = [tf.Variable(tf.random_uniform([self.cuts])) for i in xrange(self.X_train.shape[1])]

        seed = 1990

        leaf_score = tf.Variable(tf.random_uniform([(self.cuts+1)**self.X_train.shape[1], self.y_train.shape[1]]))

        x_ph = tf.placeholder(tf.float32, [None, self.X_train.shape[1]])
        y_ph = tf.placeholder(tf.float32, [None, self.y_train.shape[1]])


        y_pred = self.nn_decision_tree(x_ph, cut_points_list, leaf_score)


        if self.regularizer == True:

            for cut_points in cut_points_list:
              cut_points = -tf.nn.top_k(-cut_points, self.cuts)[0]
              num_bin.append(tf.cumsum(tf.concat([tf.constant(0.0, shape=[1]), -cut_points],0)))

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_ph) +  self.beta * tf.nn.l2_loss(leaf_score) + self.beta * tf.nn.l2_loss(tf.stack(num_bin)))
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_ph))


        opt = tf.train.AdamOptimizer(0.1)
        train_step = opt.minimize(loss)

        sess = tf.InteractiveSession()
        tf.set_random_seed(1990)

        self.time = time.time()
        sess.run(tf.initialize_all_variables())

        for epoch in range(self.epochs):

          avg_cost = 0

          total_batch = int(self.X_train.shape[0]/self.batch_size)

          for i in range(total_batch):

            batch_mask = np.random.choice(self.X_train.shape[0], self.batch_size)


            batch_x = self.X_train[batch_mask].reshape(-1, self.X_train.shape[1])
            batch_y = self.y_train[batch_mask].reshape(-1, self.y_train.shape[1])


            _, loss_e = sess.run([train_step, loss], feed_dict={x_ph: batch_x, y_ph: batch_y})

            avg_cost += loss_e / total_batch

          print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)

        self.time = time.time() - self.time

        #print("--- %s seconds ---" % (self.time))

        self.train_error = 1 - np.mean(np.argmax(y_pred.eval(feed_dict={x_ph: self.X_train}), axis=1) == np.argmax(self.y_train, axis=1))
        self.test_error = 1 - np.mean(np.argmax(y_pred.eval(feed_dict={x_ph: self.X_test}), axis=1) == np.argmax(self.y_test, axis=1))

        sess.close()

    def fit_forest(self, max_features):

        num_trees = int(ceil(self.X_train.shape[1]/max_features))

        predictions = []
        accuracy = []

        self.time = time.time()

        for i in xrange(num_trees):

            bin_list = []

            list_cut_points = [tf.Variable(tf.random_uniform([self.cuts])) for i in xrange(max_features)]

            score_leaf = tf.Variable(tf.random_uniform([(self.cuts+1)**max_features, self.y_train.shape[1]]))

            features=[]

            for i in xrange(max_features):
              features.append(random.randrange(0, self.X_train.shape[1]))

            col_idx = np.array(features)

            train_x = self.X_train[:, col_idx]
            train_y = self.y_train

            test_x = self.X_test[:, col_idx]
            test_y = self.y_test

            random_set = np.random.choice(train_x.shape[0], int(ceil(train_x.shape[0]/num_trees)))
            train_xx = train_x[random_set].reshape(-1, train_x.shape[1])
            train_yy = train_y[random_set].reshape(-1, train_y.shape[1])

            seed = 1990

            x_ph = tf.placeholder(tf.float32, [None, max_features])
            y_ph = tf.placeholder(tf.float32, [None, self.y_train.shape[1]])

            y_pred = self.nn_decision_tree(x_ph, list_cut_points, score_leaf)

            pred_proba = tf.nn.softmax(y_pred)


            if self.regularizer == True:

                for cut_points in list_cut_points:
                    cut_points = -tf.nn.top_k(-cut_points, self.cuts)[0]
                    bin_list.append(tf.cumsum(tf.concat([tf.constant(0.0, shape=[1]), -cut_points],0)))

                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_ph) +  self.beta * tf.nn.l2_loss(score_leaf) + self.beta * tf.nn.l2_loss(tf.stack(bin_list)))
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_ph))

            opt = tf.train.AdamOptimizer(0.1)
            train_step = opt.minimize(loss)

            sess = tf.InteractiveSession()
            tf.set_random_seed(1990)

            sess.run(tf.initialize_all_variables())

            for epoch in range(self.epochs):

                total_batch = int(train_xx.shape[0]/self.batch_size)

                for i in range(total_batch):

                    batch_mask = np.random.choice(train_xx.shape[0], self.batch_size)


                    batch_x = train_xx[batch_mask].reshape(-1, train_xx.shape[1])
                    batch_y = train_yy[batch_mask].reshape(-1, train_yy.shape[1])


                    _, loss_e = sess.run([train_step, loss], feed_dict={x_ph: batch_x, y_ph: batch_y})

            """For each tree, the predicted probability values and the prediction accuracies are stored
            in two different lists after training each tree for 100 epochs"""

            accuracy.append(1/np.mean(np.argmax(y_pred.eval(feed_dict={x_ph: test_x}), axis=1) == np.argmax(self.y_test, axis=1)))
            predictions.append(pred_proba.eval(feed_dict={x_ph: test_x}))

            sess.close()

        self.time = time.time() - self.time

        for index, value in enumerate(self.softmax(accuracy)):
            predictions[index] = predictions[index]*value

        pred = sum(predictions)
        self.test_error = 1 - np.mean(np.argmax(pred, axis=1) == np.argmax(self.y_test, axis=1))



    def reg_tree(self):

        num_bin = []

        cut_points_list = [tf.Variable(tf.random_uniform([self.cuts])) for i in xrange(self.X_train.shape[1])]


        seed = 1990

        leaf_score = tf.Variable(tf.random_uniform([(self.cuts+1)**self.X_train.shape[1], self.y_train.shape[1]]))

        x_ph = tf.placeholder(tf.float32, [None, self.X_train.shape[1]])
        y_ph = tf.placeholder(tf.float32, [None, self.y_train.shape[1]])


        y_pred = self.nn_decision_tree(x_ph, cut_points_list, leaf_score)


        if self.regularizer == True:
            for cut_points in cut_points_list:
              cut_points = -tf.nn.top_k(-cut_points, self.cuts)[0]
              num_bin.append(tf.cumsum(tf.concat([tf.constant(0.0, shape=[1]), -cut_points],0)))

            loss = tf.reduce_mean(tf.square(y_pred - y_ph) +  self.beta * tf.nn.l2_loss(leaf_score) + self.beta * tf.nn.l2_loss(tf.stack(num_bin)))
        else:
            loss = tf.reduce_mean(tf.square(y_pred - y_ph))


        opt = tf.train.AdamOptimizer(0.1)
        train_step = opt.minimize(loss)

        sess = tf.InteractiveSession()
        tf.set_random_seed(1990)

        self.time = time.time()
        sess.run(tf.initialize_all_variables())

        for epoch in range(self.epochs):

          avg_cost = 0

          total_batch = int(self.X_train.shape[0]/self.batch_size)

          for i in range(total_batch):

            batch_mask = np.random.choice(self.X_train.shape[0], self.batch_size)


            batch_x = self.X_train[batch_mask].reshape(-1, self.X_train.shape[1])
            batch_y = self.y_train[batch_mask].reshape(-1, self.y_train.shape[1])


            _, loss_e = sess.run([train_step, loss], feed_dict={x_ph: batch_x, y_ph: batch_y})

            avg_cost += loss_e / total_batch

          print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)

        self.time = time.time() - self.time

        #print("--- %s seconds ---" % (self.time))

        self.test_error = np.sqrt(((y_pred.eval(feed_dict={x_ph: self.X_test}) - self.y_test) ** 2).mean())

        sess.close()


    def reg_forest(self, max_features):

        num_trees = int(ceil(self.X_train.shape[1]/max_features))

        predictions = []
        accuracy = []

        self.time = time.time()

        for i in xrange(num_trees):

            bin_list = []

            list_cut_points = [tf.Variable(tf.random_uniform([self.cuts])) for i in xrange(max_features)]


            score_leaf = tf.Variable(tf.random_uniform([(self.cuts+1)**max_features, self.y_train.shape[1]]))

            features=[]

            for i in xrange(max_features):
              features.append(random.randrange(0, self.X_train.shape[1]))

            col_idx = np.array(features)

            train_x = self.X_train[:, col_idx]
            train_y = self.y_train

            test_x = self.X_test[:, col_idx]
            test_y = self.y_test

            random_set = np.random.choice(train_x.shape[0], int(ceil(train_x.shape[0]/num_trees)))

            train_xx = train_x[random_set].reshape(-1, train_x.shape[1])
            train_yy = train_y[random_set].reshape(-1, train_y.shape[1])

            seed = 1990

            x_ph = tf.placeholder(tf.float32, [None, max_features])
            y_ph = tf.placeholder(tf.float32, [None, self.y_train.shape[1]])

            y_pred = self.nn_decision_tree(x_ph, list_cut_points, score_leaf)


            if self.regularizer == True:

                for cut_points in list_cut_points:
                    cut_points = -tf.nn.top_k(-cut_points, self.cuts)[0]
                    bin_list.append(tf.cumsum(tf.concat([tf.constant(0.0, shape=[1]), -cut_points],0)))

                loss = tf.reduce_mean(tf.square(y_pred - y_ph) +  self.beta * tf.nn.l2_loss(score_leaf) + self.beta * tf.nn.l2_loss(tf.stack(bin_list)))
            else:
                loss = tf.reduce_mean(tf.square(y_pred - y_ph))

            opt = tf.train.AdamOptimizer(0.1)
            train_step = opt.minimize(loss)

            sess = tf.InteractiveSession()
            tf.set_random_seed(1990)

            sess.run(tf.initialize_all_variables())

            for epoch in range(self.epochs):

                total_batch = int(train_xx.shape[0]/self.batch_size)

                for i in range(total_batch):

                    batch_mask = np.random.choice(train_xx.shape[0], self.batch_size)


                    batch_x = train_xx[batch_mask].reshape(-1, train_xx.shape[1])
                    batch_y = train_yy[batch_mask].reshape(-1, train_yy.shape[1])


                    _, loss_e = sess.run([train_step, loss], feed_dict={x_ph: batch_x, y_ph: batch_y})

            """For each tree, the predicted values and the original y_values are stacked vertically
            in two different numpy arrays after training each tree for 100 epochs"""

            accuracy.append(np.sqrt(((y_pred.eval(feed_dict={x_ph: test_x}) - test_y) ** 2).mean()))
            predictions.append(y_pred.eval(feed_dict={x_ph: test_x}))

            sess.close()

        self.time = time.time() - self.time

        for index, value in enumerate(self.softmax(accuracy)):
            predictions[index] = predictions[index]*value

        pred = sum(predictions)
        
        self.test_error = np.sqrt(((pred - self.y_test) ** 2).mean())



