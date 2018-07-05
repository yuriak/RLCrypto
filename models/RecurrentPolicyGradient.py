# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os

class RPG_CryptoNG_ShareV(object):
    def __init__(self, feature_number, action_size=1, hidden_units_number=[128, 64], learning_rate=0.001):
        tf.reset_default_graph()
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, None, feature_number], name='s')
        self.a = tf.placeholder(dtype=tf.int32, shape=[None, None, action_size], name='a')
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, None], name='r')
        self.s_next = tf.placeholder(dtype=tf.float32, shape=[None, None, feature_number], name='s_next')
        self.action_size = action_size
        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []
        self.s_next_buffer = []
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        with tf.variable_scope('rnn_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            cell = self._add_GRUs(units_number=[128] * 2, activation=[tf.nn.tanh] * 2, keep_prob=self.dropout_keep_prob)
            self.rnn_output, _ = tf.nn.dynamic_rnn(inputs=self.s, cell=cell, dtype=tf.float32)
        
        with tf.variable_scope('supervised', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            self.state_predict = self._add_dense_layer(inputs=self.rnn_output, output_shape=hidden_units_number, drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            self.state_predict = self._add_dense_layer(inputs=self.rnn_output, output_shape=[feature_number], drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
            self.state_loss = tf.losses.mean_squared_error(self.state_predict, self.s_next)
        
        with tf.variable_scope('policy_gradient', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            self.a_prob = self._add_dense_layer(inputs=self.rnn_output, output_shape=hidden_units_number, drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            self.a_prob = self._add_dense_layer(inputs=self.a_prob, output_shape=[action_size], drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
            self.a_out = tf.nn.softmax(self.a_prob, axis=-1)
            self.negative_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.a_prob, labels=self.a)
        
        with tf.variable_scope('train'):
            optimizer_rl = tf.train.AdamOptimizer(learning_rate=learning_rate)
            optimizer_sl = tf.train.AdamOptimizer(learning_rate=learning_rate * 2)
            self.rl_loss = tf.reduce_mean(self.negative_cross_entropy * self.r)
            self.sl_loss = tf.reduce_mean(self.state_loss)
            self.rl_train_op = optimizer_rl.minimize(self.rl_loss)
            self.sl_train_op = optimizer_sl.minimize(self.sl_loss)
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
    
    def _add_GRU(self, units_number, activation=tf.nn.relu, keep_prob=1.0):
        cell = tf.contrib.rnn.GRUCell(units_number, activation=activation)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
        return cell
    
    def _add_GRUs(self, units_number, activation, keep_prob=1.0):
        cells = tf.contrib.rnn.MultiRNNCell(cells=[self._add_GRU(units_number=n, activation=a) for n, a in zip(units_number, activation)])
        return cells
    
    def _add_gru_cell(self, units_number, activation=tf.nn.relu):
        return tf.contrib.rnn.GRUCell(num_units=units_number, activation=activation)
    
    def train(self, drop=0.85):
        feed = {
            self.a: np.transpose(np.array(self.a_buffer), axes=[1, 0, 2]),
            self.r: np.transpose(np.array(self.r_buffer), axes=[1, 0]),
            self.s: np.transpose(np.array(self.s_buffer), axes=[1, 0, 2]),
            self.s_next: np.transpose(np.array(self.s_next_buffer), axes=[1, 0, 2]),
            self.dropout_keep_prob: drop
        }
        self.session.run([self.rl_train_op, self.sl_train_op], feed_dict=feed)
    
    def restore_buffer(self):
        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []
        self.s_next_buffer = []
    
    def save_current_state(self, s):
        self.s_buffer.append(s)
    
    def save_transation(self, a, r, s_next):
        self.a_buffer.append(a)
        self.r_buffer.append(r)
        self.s_next_buffer.append(s_next)
    
    def trade(self, train=False, drop=1.0, prob=False):
        feed = {
            self.s: np.transpose(np.array(self.s_buffer), axes=[1, 0, 2]),
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
    
    def load_model(self, model_path='./RecurrentPolicyGradient'):
        self.saver.restore(self.session, model_path + '/model')
    
    def save_model(self, model_path='./RecurrentPolicyGradient'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/model'
        self.saver.save(self.session, model_file)