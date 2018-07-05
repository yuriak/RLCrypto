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


class RPG_Crypto_portfolio(object):
    def __init__(self, feature_number, action_size=1, hidden_units_number=[128, 64], learning_rate=0.001):
        tf.reset_default_graph()
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, feature_number], name='s')
        self.a = tf.placeholder(dtype=tf.int32, shape=[None, action_size], name='a')
        self.r = tf.placeholder(dtype=tf.float32, shape=[None], name='r')
        self.s_next = tf.placeholder(dtype=tf.float32, shape=[None, feature_number], name='s_next')
        self.action_size = action_size
        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []
        self.s_next_buffer = []
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        with tf.variable_scope('rnn_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            cell = self._add_GRU(units_number=128, activation=tf.nn.tanh, keep_prob=self.dropout_keep_prob)
            #             cells=self._add_GRUs(units_number=[256,128],activation=[tf.nn.relu,tf.nn.tanh])
            self.rnn_input = tf.expand_dims(self.s, axis=0)
            self.rnn_output, _ = tf.nn.dynamic_rnn(inputs=self.rnn_input, cell=cell, dtype=tf.float32)
            #             self.rnn_output=tf.contrib.layers.layer_norm(self.rnn_output)
            self.rnn_output = tf.unstack(self.rnn_output, axis=0)[0]
        
        with tf.variable_scope('supervised', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            self.state_predict = self._add_dense_layer(inputs=self.rnn_output, output_shape=hidden_units_number, drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            #             self.state_predict=tf.contrib.layers.layer_norm(self.state_predict)
            self.state_predict = self._add_dense_layer(inputs=self.rnn_output, output_shape=[feature_number], drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
            self.state_loss = tf.losses.mean_squared_error(self.state_predict, self.s_next)
        
        with tf.variable_scope('policy_gradient', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            #             self.rnn_output=tf.stop_gradient(self.rnn_output)
            self.a_prob = self._add_dense_layer(inputs=self.rnn_output, output_shape=hidden_units_number, drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            #             self.a_prob=tf.contrib.layers.layer_norm(self.a_prob)
            self.a_prob = self._add_dense_layer(inputs=self.a_prob, output_shape=[action_size], drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
            self.a_out = tf.nn.softmax(self.a_prob, axis=-1)
            self.negative_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.a_prob, labels=self.a)
        
        with tf.variable_scope('train'):
            optimizer_rl = tf.train.AdamOptimizer(learning_rate=learning_rate)
            optimizer_sl = tf.train.AdamOptimizer(learning_rate=learning_rate * 2)
            self.rlloss = tf.reduce_mean(self.negative_cross_entropy * self.r)
            self.slloss = tf.reduce_mean(self.state_loss)
            self.rltrain_op = optimizer_rl.minimize(self.rlloss)
            self.sltrain_op = optimizer_sl.minimize(self.slloss)
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
        #         np.random.shuffle(random_index)
        feed = {
            self.a: np.array(self.a_buffer),
            self.r: np.array(self.r_buffer),
            self.s: np.array(self.s_buffer),
            self.s_next: np.array(self.s_next_buffer),
            self.dropout_keep_prob: drop
        }
        self.session.run([self.rltrain_op, self.sltrain_op], feed_dict=feed)
    
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
    
    def trade(self, s, train=False, drop=1.0, prob=False):
        feed = {
            self.s: np.array(self.s_buffer),
            self.dropout_keep_prob: drop
        }
        a_prob = self.session.run([self.a_out], feed_dict=feed)
        a_prob = a_prob[-1][-1].flatten()
        if train:
            a_indices = np.arange(a_prob.shape[0])
            target_index = np.random.choice(a_indices, p=a_prob)
            a = np.zeros(a_prob.shape[0])
            a[target_index] = 1.0
            return a
        else:
            if prob:
                return a_prob
            target_index = np.argmax(a_prob)
            a = np.zeros(a_prob.shape[0])
            a[target_index] = 1.0
            return a
    
    def load_model(self, model_path='./RPGModel'):
        self.saver.restore(self.session, model_path + '/model')
    
    def save_model(self, model_path='./RPGModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/model'
        self.saver.save(self.session, model_file)


class DRL_Crypto_portfolio(object):
    def __init__(self, feature_number, action_size=1, c=1e-5, hidden_units_number=[128, 64], learning_rate=0.001):
        tf.reset_default_graph()
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, feature_number], name='s')
        self.d = tf.placeholder(dtype=tf.float32, shape=[None, action_size - 1], name='d')
        self.s_buffer = []
        self.d_buffer = []
        self.c = c
        self.action_size = action_size
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        with tf.variable_scope('rnn_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            #             cell=self._add_GRU(units_number=128,keep_prob=self.dropout_keep_prob)
            cells = self._add_GRUs(units_number=[128, action_size], activation=[tf.nn.relu, tf.nn.relu])
            self.rnn_input = tf.expand_dims(self.s, axis=0)
            self.rnn_output, _ = tf.nn.dynamic_rnn(inputs=self.rnn_input, cell=cells, dtype=tf.float32)
            #             self.rnn_output=tf.contrib.layers.layer_norm(self.rnn_output)
            self.a_prob = tf.unstack(self.rnn_output, axis=0)[0]
        
        with tf.variable_scope('direct_RL', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            #             self.rnn_output=tf.stop_gradient(self.rnn_output)
            #             self.a_prob = self._add_dense_layer(inputs=self.rnn_output, output_shape=hidden_units_number+[action_size], drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            #             self.a_prob = self._add_dense_layer(inputs=self.a_prob, output_shape=, drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
            self.a_out = tf.nn.softmax(self.a_prob, axis=-1)
            self.a_out = tf.concat((tf.zeros(dtype=tf.float32, shape=[1, self.action_size]), self.a_out), axis=0)
            self.reward = tf.reduce_sum(self.d * self.a_out[:-1, :-1] - self.c * tf.abs(self.a_out[1:, :-1] - self.a_out[:-1, :-1]), axis=1)
            self.total_reward = tf.reduce_sum(self.reward)
            self.mean_reward = tf.reduce_mean(self.reward)
        
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(-self.mean_reward)
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
        cell = tf.contrib.rnn.LSTMCell(units_number, activation=activation)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
        return cell
    
    def _add_GRUs(self, units_number, activation, keep_prob=1.0):
        cells = tf.contrib.rnn.MultiRNNCell(cells=[self._add_GRU(units_number=n, activation=a) for n, a in zip(units_number, activation)])
        return cells
    
    def _add_gru_cell(self, units_number, activation=tf.nn.relu):
        return tf.contrib.rnn.GRUCell(num_units=units_number, activation=activation)
    
    def train(self, drop=0.85):
        #         np.random.shuffle(random_index)
        feed = {
            self.s: np.array(self.s_buffer),
            self.d: np.array(self.d_buffer),
            self.dropout_keep_prob: drop
        }
        self.session.run([self.train_op], feed_dict=feed)
    
    def restore_buffer(self):
        self.s_buffer = []
        self.d_buffer = []
    
    def save_current_state(self, s, d):
        self.s_buffer.append(s)
        self.d_buffer.append(d)
    
    def trade(self, train=False, drop=1.0, prob=False):
        feed = {
            self.s: np.array(self.s_buffer),
            self.dropout_keep_prob: drop
        }
        a_prob = self.session.run([self.a_out], feed_dict=feed)
        a_prob = a_prob[-1][-1].flatten()
        return a_prob
    
    def load_model(self, model_path='./RPGModel'):
        self.saver.restore(self.session, model_path + '/model')
    
    def save_model(self, model_path='./RPGModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/model'
        self.saver.save(self.session, model_file)


class DuelingDQN_portfolio(object):
    def __init__(self, a_dim, s_dim, buffer_size, batch_size, update_target_interval=50, epsilon=0.9, gamma=0.9, learning_rate=1e-3):
        tf.reset_default_graph()
        self.a_dim, self.s_dim = a_dim, s_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = np.zeros((self.buffer_size, self.s_dim * 2 + 2), dtype=np.float32)
        self.buffer_length = 0
        self.update_target_interval = update_target_interval
        self.critic_loss = 0
        self.total_step = 0
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.s = tf.placeholder(tf.float32, [None, self.s_dim], name='s')
        self.s_next = tf.placeholder(tf.float32, [None, self.s_dim], name='s_next')
        self.q_next = tf.placeholder(tf.float32, [None, self.a_dim], name='q_next')
        
        with tf.variable_scope('q_eval', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)) as scope:
            self.q_eval = self._build_net(self.s, scope)
        
        with tf.variable_scope('q_target', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)) as scope:
            self.q_target = self._build_net(self.s_next, scope)
        
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_next, self.q_eval))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_eval')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target')
            self.update_q_target_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        self.init_op = tf.global_variables_initializer()
        self.session = tf.Session()
        self.saver = tf.train.Saver()
    
    def init_model(self):
        self.session.run(self.init_op)
    
    def _build_net(self, s, scope):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 64, activation=tf.nn.tanh, name='l1')
            net = tf.layers.dense(net, 32, activation=tf.nn.tanh, name='l2')
            value = tf.layers.dense(net, 1, activation=None, name='a')
            advantage = tf.layers.dense(net, self.a_dim, activation=None, name='advantage')
            q = value + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True))
            return q
    
    def trade(self, s, train=False):
        q = self.session.run(self.q_eval, {self.s: s})
        a = np.argmax(q)
        action = np.zeros(self.a_dim)
        if train:
            if np.random.uniform() < self.epsilon:
                action[a] = 1.0
                return action
            else:
                action[np.random.randint(0, self.a_dim)] = 1.0
                return action
        action[a] = 1.0
        return action
    
    def update_target(self):
        self.session.run(self.update_q_target_op)
    
    def train(self):
        if self.buffer_length < self.buffer_size:
            return
        if self.total_step % self.update_target_interval == 0:
            self.session.run(self.update_q_target_op)
        s, a, r, s_next = self.get_transition_batch()
        q_eval, q_target = self.session.run([self.q_eval, self.q_target], {self.s: s, self.s_next: s_next})
        b_indices = np.arange(self.batch_size, dtype=np.int32)
        q_next = q_eval.copy()
        q_next[b_indices, a.astype(np.int)] = r + self.gamma * np.max(q_target, axis=1)
        _, self.critic_loss = self.session.run([self.train_op, self.loss], {self.s: s, self.q_next: q_next})
        self.total_step += 1
    
    def save_transition(self, s, a, r, s_next):
        a = np.argmax(a)
        transition = np.hstack((s, [a], [r], s_next))
        self.buffer[self.buffer_length % self.buffer_size, :] = transition
        self.buffer_length += 1
    
    def get_transition_batch(self):
        indices = np.random.choice(self.buffer_size, size=self.batch_size)
        batch = self.buffer[indices, :]
        s = batch[:, :self.s_dim]
        a = batch[:, self.s_dim: self.s_dim + 1]
        r = batch[:, -self.s_dim - 1: -self.s_dim]
        s_next = batch[:, -self.s_dim:]
        return s, a, r, s_next
    
    def restore_buffer(self):
        self.buffer = np.zeros((self.buffer_size, self.s_dim + 1 + 1 + self.s_dim))
        self.buffer_length = 0
    
    def load_model(self, model_path='./DRLModel'):
        self.saver.restore(self.session, model_path + '/model')
    
    def save_model(self, model_path='./DRLModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/model'
        self.saver.save(self.session, model_file)


class RPG_CryptoNG(object):
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
            cell = self._add_GRU(units_number=128, activation=tf.nn.tanh, keep_prob=self.dropout_keep_prob)
            #             self.rnn_input=tf.expand_dims(self.s,axis=0)
            self.rnn_output, _ = tf.nn.dynamic_rnn(inputs=self.s, cell=cell, dtype=tf.float32)
        # self.rnn_output=tf.unstack(self.rnn_output,axis=0)[0]
        
        with tf.variable_scope('supervised', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            self.state_predict = self._add_dense_layer(inputs=self.rnn_output, output_shape=hidden_units_number, drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            self.state_predict = self._add_dense_layer(inputs=self.rnn_output, output_shape=[feature_number], drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
            self.state_loss = tf.losses.mean_squared_error(self.state_predict, self.s_next)
        
        with tf.variable_scope('policy_gradient', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            #             self.rnn_output=tf.stop_gradient(self.rnn_output)
            self.a_prob = self._add_dense_layer(inputs=self.rnn_output, output_shape=hidden_units_number, drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            self.a_prob = self._add_dense_layer(inputs=self.a_prob, output_shape=[action_size], drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
            self.a_out = tf.nn.softmax(self.a_prob, axis=-1)
            self.negative_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.a_prob, labels=self.a)
        
        with tf.variable_scope('train'):
            optimizer_rl = tf.train.AdamOptimizer(learning_rate=learning_rate)
            optimizer_sl = tf.train.AdamOptimizer(learning_rate=learning_rate * 2)
            self.rlloss = tf.reduce_mean(self.negative_cross_entropy * self.r)
            self.slloss = tf.reduce_mean(self.state_loss)
            self.rltrain_op = optimizer_rl.minimize(self.rlloss)
            self.sltrain_op = optimizer_sl.minimize(self.slloss)
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
        #         np.random.shuffle(random_index)
        feed = {
            self.a: np.transpose(np.array(self.a_buffer), axes=[1, 0, 2]),
            self.r: np.transpose(np.array(self.r_buffer), axes=[1, 0]),
            self.s: np.transpose(np.array(self.s_buffer), axes=[1, 0, 2]),
            self.s_next: np.transpose(np.array(self.s_next_buffer), axes=[1, 0, 2]),
            self.dropout_keep_prob: drop
        }
        self.session.run([self.rltrain_op, self.sltrain_op], feed_dict=feed)
    
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
        #         print(np.array(self.a_buffer).shape)
        #         print(np.transpose(np.array(self.a_buffer),axes=[1,0,2]).shape)
        feed = {
            self.s: np.transpose(np.array(self.s_buffer), axes=[1, 0, 2]),
            self.dropout_keep_prob: drop
        }
        a_prob = self.session.run([self.a_out], feed_dict=feed)[0]
        
        a_prob = a_prob[:, -1, :]
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
    
    def load_model(self, model_path='./RPGModel'):
        self.saver.restore(self.session, model_path + '/model')
    
    def save_model(self, model_path='./RPGModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/model'
        self.saver.save(self.session, model_file)


class RPG_Portfolio_Stable(object):
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
            cell = self._add_GRU(units_number=128, activation=tf.nn.tanh, keep_prob=self.dropout_keep_prob)
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
    
    def trade(self, train=False, kp=1.0, prob=False):
        feed = {
            self.s: np.transpose(np.array(self.s_buffer), axes=[1, 0, 2]),
            self.dropout_keep_prob: kp
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
    
    def load_model(self, model_path='./RPGModel'):
        self.saver.restore(self.session, model_path + '/model')
    
    def save_model(self, model_path='./RPGModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/model'
        self.saver.save(self.session, model_file)


class DDRPG(object):
    def __init__(self, s_dim,
                 asset_number,
                 buffer_size=1600,
                 batch_size=64,
                 tau=0.05,
                 softmax_tau=1,
                 gamma=0.99,
                 actor_rnn_units=128,
                 critic_rnn_units=128,
                 actor_dnn_units=[64],
                 critic_dnn_units=[64],
                 learning_rate_a=1e-3,
                 learning_rate_c=2e-3):
        tf.reset_default_graph()
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.pointer = 0
        self.asset_number = asset_number
        
        self.a_dim, self.s_dim = 1, s_dim
        self.tau = tau
        self.softmax_tau = softmax_tau
        self.gamma = gamma
        self.lr_a = learning_rate_a
        self.lr_c = learning_rate_c
        self.actor_rnn_units = actor_rnn_units
        self.critic_rnn_units = critic_rnn_units
        self.actor_dnn_units = actor_dnn_units
        self.critic_dnn_units = critic_dnn_units
        
        self.s_buffer = np.zeros((self.asset_number, self.buffer_size, self.s_dim))
        self.s_next_buffer = np.zeros((self.asset_number, self.buffer_size, self.s_dim))
        self.r_buffer = np.zeros((self.asset_number, self.buffer_size, 1))
        self.a_buffer = np.zeros((self.asset_number, self.buffer_size, self.a_dim))
        
        self.s = tf.placeholder(tf.float32, [None, None, self.s_dim], 's')
        self.s_next = tf.placeholder(tf.float32, [None, None, self.s_dim], 's_next')
        self.r = tf.placeholder(tf.float32, [None, None, 1], 'r')
        self.keep_prob = tf.placeholder(tf.float32, [], 'dropout')
        #         with tf.variable_scope('actor', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
        with tf.variable_scope('actor', initializer=tf.truncated_normal_initializer(dtype=tf.float32, mean=0, stddev=1)):
            self.a = self._build_a(s=self.s,
                                   rnn_units=self.actor_rnn_units,
                                   dnn_units=self.actor_dnn_units,
                                   scope='predict',
                                   keep_prob=self.keep_prob,
                                   trainable=True)
            a_next = self._build_a(s=self.s_next,
                                   rnn_units=self.actor_rnn_units,
                                   dnn_units=self.actor_dnn_units,
                                   scope='target',
                                   keep_prob=self.keep_prob,
                                   trainable=False)
        # with tf.variable_scope('critic', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
        with tf.variable_scope('critic', initializer=tf.truncated_normal_initializer(dtype=tf.float32, mean=0, stddev=1)):
            q = self._build_c(s=self.s,
                              a=self.a,
                              rnn_units=self.critic_rnn_units,
                              dnn_units=self.critic_dnn_units,
                              scope='predict',
                              keep_prob=self.keep_prob,
                              trainable=True)
            q_next = self._build_c(s=self.s_next,
                                   a=a_next,
                                   rnn_units=self.critic_rnn_units,
                                   dnn_units=self.critic_dnn_units,
                                   scope='target',
                                   keep_prob=self.keep_prob,
                                   trainable=False)
        
        self.ap_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/predict')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/target')
        self.cp_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/predict')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/target')
        
        self.soft_replace = [[tf.assign(ta, (1 - self.tau) * ta + self.tau * pa),
                              tf.assign(tc, (1 - self.tau) * tc + self.tau * pc)]
                             for ta, pa, tc, pc in zip(self.at_params, self.ap_params, self.ct_params, self.cp_params)]
        
        q_target = self.r + self.gamma * q_next
        with tf.variable_scope('actor_loss'):
            a_loss = - tf.reduce_mean(q)
        with tf.variable_scope('critic_loss'):
            c_loss = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        with tf.variable_scope('actor_train'):
            self.a_train = tf.train.AdamOptimizer(self.lr_a).minimize(a_loss, var_list=self.ap_params)
        with tf.variable_scope('critic_loss'):
            self.c_train = tf.train.AdamOptimizer(self.lr_c).minimize(c_loss, var_list=self.cp_params)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def trade(self, train=False, kp=1.0, epsilon=0.9):
        start_point = self.pointer + 1 - self.batch_size
        action = self.sess.run(self.a, {self.s: self.s_buffer[:, 0 if start_point < 0 else start_point:self.pointer + 1, :], self.keep_prob: kp})[:, -1, :].flatten()
        if train:
            if np.random.rand() < epsilon:
                return np.exp(action / self.softmax_tau) / np.sum(np.exp(action / self.softmax_tau))
            action = np.random.normal(action, scale=(1 - action))
            action = np.exp(action / self.softmax_tau) / np.sum(np.exp(action / self.softmax_tau))
            return action
        else:
            return np.exp(action / self.softmax_tau) / np.sum(np.exp(action / self.softmax_tau))
    
    def train(self, kp=0.85):
        if self.pointer < self.batch_size:
            return
        sample_start = np.random.randint(0, self.pointer - self.batch_size + 1)
        self.sess.run(self.soft_replace)
        self.sess.run(self.a_train, {self.s: self.s_buffer[:, sample_start:sample_start + self.batch_size],
                                     self.keep_prob: kp})
        self.sess.run(self.c_train, {self.s: self.s_buffer[:, sample_start:sample_start + self.batch_size],
                                     self.a: self.a_buffer[:, sample_start:sample_start + self.batch_size],
                                     self.r: self.r_buffer[:, sample_start:sample_start + self.batch_size],
                                     self.s_next: self.s_next_buffer[:, sample_start:sample_start + self.batch_size],
                                     self.keep_prob: kp})
    
    def save_current_state(self, s):
        self.s_buffer[:, self.pointer, :] = s
    
    def save_transition(self, a, r, s_next):
        self.a_buffer[:, self.pointer, :] = a[:, None]
        self.s_next_buffer[:, self.pointer, :] = s_next
        self.r_buffer[:, self.pointer, :] = r[:, None]
    
    def settle(self):
        self.pointer += 1
    
    def restore_buffer(self):
        self.s_buffer = np.zeros((self.asset_number, self.buffer_size, self.s_dim))
        self.s_next_buffer = np.zeros((self.asset_number, self.buffer_size, self.s_dim))
        self.r_buffer = np.zeros((self.asset_number, self.buffer_size, 1))
        self.a_buffer = np.zeros((self.asset_number, self.buffer_size, self.a_dim))
        self.pointer = 0
    
    def _build_a(self, s, rnn_units, dnn_units, scope, keep_prob, trainable):
        with tf.variable_scope(scope):
            cell = self._add_GRU(units_number=rnn_units, activation=tf.nn.tanh, keep_prob=keep_prob, trainable=trainable)
            out, _ = tf.nn.dynamic_rnn(inputs=s, cell=cell, dtype=tf.float32)
            out = self._add_dense_layer(inputs=out, output_shape=dnn_units + [1], activations=([tf.nn.relu] * (len(dnn_units)) + [tf.nn.sigmoid]), drop_keep_prob=keep_prob, trainable=trainable)
            #             out = out / self.softmax_tau
            #             out = tf.nn.softmax(out, axis=0)
            return out
    
    def _build_c(self, s, a, rnn_units, dnn_units, scope, keep_prob, trainable):
        with tf.variable_scope(scope):
            rnn_input = tf.concat([s, a], axis=-1)
            cell = self._add_GRU(units_number=rnn_units, activation=tf.nn.tanh, keep_prob=keep_prob, trainable=trainable)
            out, _ = tf.nn.dynamic_rnn(inputs=rnn_input, cell=cell, dtype=tf.float32)
            out = self._add_dense_layer(inputs=out, output_shape=dnn_units, activations=[tf.nn.relu] * len(dnn_units), drop_keep_prob=keep_prob, trainable=trainable)
            q = self._add_dense_layer(inputs=out, output_shape=[1], activations=[tf.nn.tanh], drop_keep_prob=keep_prob, trainable=trainable)
            return q
    
    def _add_GRU(self, units_number, activation=tf.nn.relu, keep_prob=1.0, trainable=True):
        cell = tf.contrib.rnn.GRUCell(units_number, activation=activation)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
        return cell
    
    def _add_dense_layer(self, inputs, output_shape, activations, drop_keep_prob, use_bias=True, trainable=True):
        output = inputs
        for n, a in zip(output_shape, activations):
            output = tf.layers.dense(output, n, activation=a, use_bias=use_bias, trainable=trainable)
            output = tf.nn.dropout(output, drop_keep_prob)
        return output


class RPG_CryptoNG_ShareVNG(object):
    def __init__(self, s_dim, asset_number, a_dim, buffer_size=64, batch_size=64, hidden_units_number=[128, 64], learning_rate=0.001):
        tf.reset_default_graph()
        self.a_dim = a_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.asset_number = asset_number
        self.s_dim = s_dim
        self.pointer = 0
        self.a_buffer = np.zeros((self.asset_number, self.buffer_size, self.a_dim))
        self.r_buffer = np.zeros((self.asset_number, self.buffer_size))
        self.s_buffer = np.zeros((self.asset_number, self.buffer_size, self.s_dim))
        self.s_next_buffer = np.zeros((self.asset_number, self.buffer_size, self.s_dim))
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, None, self.s_dim], name='s')
        self.a = tf.placeholder(dtype=tf.int32, shape=[None, None, self.a_dim], name='a')
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, None], name='r')
        self.s_next = tf.placeholder(dtype=tf.float32, shape=[None, None, s_dim], name='s_next')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        with tf.variable_scope('rnn_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            cell = self._add_GRU(units_number=128, activation=tf.nn.tanh, keep_prob=self.dropout_keep_prob)
            self.rnn_output, _ = tf.nn.dynamic_rnn(inputs=self.s, cell=cell, dtype=tf.float32)
        
        with tf.variable_scope('supervised', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            self.state_predict = self._add_dense_layer(inputs=self.rnn_output, output_shape=hidden_units_number, drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            self.state_predict = self._add_dense_layer(inputs=self.rnn_output, output_shape=[s_dim], drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
            self.state_loss = tf.losses.mean_squared_error(self.state_predict, self.s_next)
        
        with tf.variable_scope('policy_gradient', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            self.a_prob = self._add_dense_layer(inputs=self.rnn_output, output_shape=hidden_units_number, drop_keep_prob=self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            self.a_prob = self._add_dense_layer(inputs=self.a_prob, output_shape=[2], drop_keep_prob=self.dropout_keep_prob, act=None, use_bias=True)
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
    
    def train(self, kp=0.85):
        if self.pointer < self.buffer_size - 1:
            return
        max_upper_bound = self.buffer_size - self.batch_size + 1
        lower = np.random.randint(low=0, high=max_upper_bound)
        upper = lower + +self.batch_size
        
        feed = {
            self.a: self.a_buffer[:, lower:upper, :],
            self.r: self.r_buffer[:, lower:upper],
            self.s: self.s_buffer[:, lower:upper, :],
            self.s_next: self.s_next_buffer[:, lower:upper, :],
            self.dropout_keep_prob: kp
        }
        self.session.run([self.rl_train_op, self.sl_train_op], feed_dict=feed)
    
    def restore_buffer(self):
        self.a_buffer = np.zeros((self.asset_number, self.buffer_size, self.a_dim))
        self.r_buffer = np.zeros((self.asset_number, self.buffer_size))
        self.s_buffer = np.zeros((self.asset_number, self.buffer_size, self.s_dim))
        self.s_next_buffer = np.zeros((self.asset_number, self.buffer_size, self.s_dim))
        self.pointer = 0
    
    def save_current_state(self, s):
        if self.pointer < self.buffer_size - 1:
            self.s_buffer[:, self.pointer, :] = s
        else:
            self.s_buffer[:, :-1, :] = self.s_buffer[:, 1:, :]
            self.s_buffer[:, -1, :] = s
    
    def save_transition(self, a, r, s_next):
        if self.pointer < self.buffer_size - 1:
            self.a_buffer[:, self.pointer, :] = a
            self.s_next_buffer[:, self.pointer, :] = s_next
            self.r_buffer[:, self.pointer] = r
            self.pointer += 1
        else:
            self.a_buffer[:, :-1, :] = self.a_buffer[:, 1:, :]
            self.a_buffer[:, -1, :] = a
            
            self.r_buffer[:, :-1] = self.r_buffer[:, 1:]
            self.r_buffer[:, -1] = r
            
            self.s_next_buffer[:, :-1, :] = self.s_next_buffer[:, 1:, :]
            self.s_next_buffer[:, -1, :] = s_next
    
    def trade(self, train=False, kp=1.0, prob=False):
        feed = {
            self.s: self.s_buffer,
            self.dropout_keep_prob: kp
        }
        a_prob = self.session.run(self.a_out, feed_dict=feed)[:, self.pointer, :]
        action = []
        if train:
            for ap in a_prob:
                if prob:
                    action.append(np.clip(np.random.normal(ap, 0.5), 0, 1))
                else:
                    a_indices = np.arange(ap.shape[0])
                    target_index = np.random.choice(a_indices, p=ap)
                    a = np.zeros(ap.shape[0])
                    a[target_index] = 1.0
                    action.append(a)
            return np.array(action)
        else:
            if prob:
                return a_prob
            action = []
            for ap in a_prob:
                target_index = np.argmax(ap)
                a = np.zeros(ap.shape[0])
                a[target_index] = 1.0
                action.append(a)
            return np.array(action)
    
    def load_model(self, model_path='./RPGModel'):
        self.saver.restore(self.session, model_path + '/model')
    
    def save_model(self, model_path='./RPGModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/model'
        self.saver.save(self.session, model_file)
