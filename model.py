# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os


class PG_Crypto(object):
    def __init__(self, feature_number, hidden_units_number=[64, 32, 2], learning_rate=0.001):
        tf.reset_default_graph()
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, feature_number], name='environment_features')
        self.a = tf.placeholder(dtype=tf.int32, shape=[None], name='a')
        self.r = tf.placeholder(dtype=tf.float32, shape=[None], name='r')
        
        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        with tf.variable_scope('rnn', initializer=tf.contrib.layers.xavier_initializer(uniform=False), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            self.a_prob = self._add_dense_layer(inputs=self.s, output_shape=hidden_units_number, drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            self.a_out = tf.nn.softmax(self.a_prob)
        with tf.variable_scope('reward'):
            negative_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.a_prob, labels=self.a)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.loss = tf.reduce_mean(negative_cross_entropy * self.r)
            self.train_op = optimizer.minimize(self.loss)
        self.init_op = tf.global_variables_initializer()
        self.session = tf.Session()
        self.saver = tf.train.Saver()
    
    def init_model(self):
        self.session.run(self.init_op)
    
    def _add_dense_layer(self, inputs, output_shape, drop_keep_prob, act=tf.nn.relu, use_bias=True):
        output = inputs
        for n in output_shape:
            output = tf.layers.dense(output, n, activation=act, use_bias=use_bias)
            output = tf.nn.dropout(output, drop_keep_prob)
        return output
    
    def _add_gru_cell(self, units_number, activation=tf.nn.relu):
        return tf.contrib.rnn.GRUCell(num_units=units_number, activation=activation)
    
    def train(self, drop=0.85):
        feed = {
            self.a: np.array(self.a_buffer),
            self.r: np.array(self.r_buffer),
            self.s: np.array(self.s_buffer),
            self.dropout_keep_prob: drop
        }
        _, loss = self.session.run([self.train_op, self.loss], feed_dict=feed)
        return loss
    
    def restore_buffer(self):
        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []
    
    def save_transation(self, s, a, r):
        self.a_buffer.append(a)
        self.r_buffer.append(r)
        self.s_buffer.append(s)
    
    def trade(self, s, train=False, drop=1.0):
        feed = {
            self.s: s,
            self.dropout_keep_prob: drop
        }
        a_prob = self.session.run([self.a_out], feed_dict=feed)
        a_prob = a_prob[0].flatten()
        if train:
            a_indices = np.arange(a_prob.shape[0])
            return np.random.choice(a_indices, p=a_prob)
        else:
            return np.argmax(a_prob)
    
    def load_model(self, model_path='./PGModel'):
        self.saver.restore(self.session, model_path + '/model')
    
    def save_model(self, model_path='./PGModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/model'
        self.saver.save(self.session, model_file)


class PG_Crypto_portfolio(object):
    def __init__(self, feature_number, action_size=1, hidden_units_number=[300, 300, 128], learning_rate=0.001):
        tf.reset_default_graph()
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, feature_number], name='s')
        self.a = tf.placeholder(dtype=tf.int32, shape=[None, action_size], name='a')
        self.r = tf.placeholder(dtype=tf.float32, shape=[None], name='r')
        self.action_size = action_size
        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        with tf.variable_scope('policy', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            self.a_prob = self._add_dense_layer(inputs=self.s, output_shape=hidden_units_number, drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            self.a_prob = self._add_dense_layer(inputs=self.a_prob, output_shape=[self.action_size], drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
            self.a_out = tf.nn.softmax(self.a_prob)
        with tf.variable_scope('reward'):
            negative_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.a_prob, labels=self.a)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.loss = tf.reduce_mean(negative_cross_entropy * self.r)
            self.train_op = optimizer.minimize(self.loss)
        self.init_op = tf.global_variables_initializer()
        self.session = tf.Session()
        self.saver = tf.train.Saver()
    
    def init_model(self):
        self.session.run(self.init_op)
    
    def _add_dense_layer(self, inputs, output_shape, drop_keep_prob, act=tf.nn.relu, use_bias=True):
        output = inputs
        for n in output_shape:
            output = tf.layers.dense(output, n, activation=act, use_bias=use_bias)
            output = tf.nn.dropout(output, drop_keep_prob)
        return output
    
    def train(self, drop=0.85):
        random_index = np.arange(len(self.a_buffer))
        np.random.shuffle(random_index)
        feed = {
            self.a: np.array(self.a_buffer)[random_index],
            self.r: np.array(self.r_buffer)[random_index],
            self.s: np.array(self.s_buffer)[random_index],
            self.dropout_keep_prob: drop
        }
        _, loss = self.session.run([self.train_op, self.loss], feed_dict=feed)
        return loss
    
    def restore_buffer(self):
        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []
    
    def save_transation(self, s, a, r):
        self.a_buffer.append(a)
        self.r_buffer.append(r)
        self.s_buffer.append(s)
    
    def trade(self, s, train=False, drop=1.0):
        feed = {
            self.s: s,
            self.dropout_keep_prob: drop
        }
        a_prob = self.session.run([self.a_out], feed_dict=feed)
        a_prob = a_prob[0].flatten()
        if train:
            a_indices = np.arange(a_prob.shape[0])
            target_index = np.random.choice(a_indices, p=a_prob)
            a = np.zeros(a_prob.shape[0])
            a[target_index] = 1.0
            return a
        else:
            target_index = np.argmax(a_prob)
            a = np.zeros(a_prob.shape[0])
            a[target_index] = 1.0
            return a
    
    def load_model(self, model_path='./PGModel'):
        self.saver.restore(self.session, model_path + '/model')
    
    def save_model(self, model_path='./PGModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/model'
        self.saver.save(self.session, model_file)
