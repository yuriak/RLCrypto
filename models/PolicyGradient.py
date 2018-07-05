# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os


class PolicyGradient(object):
    def __init__(self, feature_number, action_size=2, hidden_units_number=[128, 128, 128, 64], learning_rate=0.001):
        tf.reset_default_graph()
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, None, feature_number], name='s')
        self.a = tf.placeholder(dtype=tf.int32, shape=[None, None, action_size], name='a')
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, None], name='r')
        self.action_size = action_size
        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        with tf.variable_scope('policy', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            self.a_prob = self._add_dense_layer(inputs=self.s, output_shape=hidden_units_number, drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            self.a_prob = self._add_dense_layer(inputs=self.a_prob, output_shape=[self.action_size], drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
            self.a_out = tf.nn.softmax(self.a_prob, axis=-1)
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
        random_index = np.arange(len(self.s_buffer))
        np.random.shuffle(random_index)
        feed = {
            self.a: np.transpose(np.array(self.a_buffer)[random_index], axes=[1, 0, 2]),
            self.r: np.transpose(np.array(self.r_buffer)[random_index], axes=[1, 0]),
            self.s: np.transpose(np.array(self.s_buffer)[random_index], axes=[1, 0, 2]),
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
    
    def trade(self, s, train=False, drop=1.0, prob=False):
        feed = {
            self.s: s[:, None, :],
            self.dropout_keep_prob: drop
        }
        a_prob = self.session.run(self.a_out, feed_dict=feed)[:, -1, :]
        actions = []
        if train:
            for ap in a_prob:
                if prob:
                    np.clip(np.random.normal(0.5, 0.25), 0, 1)
                else:
                    a_indices = np.arange(ap.shape[0])
                    target_index = np.random.choice(a_indices, p=ap)
                    a = np.zeros(ap.shape[0])
                    a[target_index] = 1.0
                    actions.append(a)
            return np.array(actions)
        else:
            if prob:
                return a_prob
            actions = []
            for ap in a_prob:
                target_index = np.argmax(ap)
                a = np.zeros(ap.shape[0])
                a[target_index] = 1.0
                actions.append(a)
            return np.array(actions)
    
    def load_model(self, model_path='./PolicyGradient'):
        self.saver.restore(self.session, model_path + '/model')
    
    def save_model(self, model_path='./PolicyGradient'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/model'
        self.saver.save(self.session, model_file)