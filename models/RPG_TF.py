# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from models.Model import Model
from models.layers import *


class RPG_TF(Model):
    def __init__(self, s_dim, b_dim, a_dim=2, hidden_units_number=[128, 64], rnn_units_number=[128, 128], learning_rate=0.001, batch_size=64, normalize_length=10):
        super(RPG_TF, self).__init__()
        tf.reset_default_graph()
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, None, s_dim], name='s')
        self.a = tf.placeholder(dtype=tf.int32, shape=[None, None, a_dim], name='a')
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, None], name='r')
        self.s_next = tf.placeholder(dtype=tf.float32, shape=[None, None, s_dim], name='s_next')
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.b_dim = b_dim
        self.batch_size = batch_size
        self.normalize_length = normalize_length
        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []
        self.s_next_buffer = []
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        with tf.variable_scope('rnn_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            cells = add_GRUs(units_numbers=rnn_units_number, acts=[tf.nn.tanh, tf.nn.tanh], kp=self.dropout_keep_prob)
            self.rnn_output, _ = tf.nn.dynamic_rnn(inputs=self.s, cell=cells, dtype=tf.float32)
        
        with tf.variable_scope('supervised', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            self.state_predict = add_dense(inputs=self.rnn_output,
                                           units_numbers=[self.s_dim],
                                           acts=([None]),
                                           kp=self.dropout_keep_prob,
                                           use_bias=True)
            self.state_loss = tf.losses.mean_squared_error(self.state_predict, self.s_next)
        
        with tf.variable_scope('policy_gradient', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            self.a_prob = add_dense(inputs=self.rnn_output,
                                    units_numbers=(hidden_units_number + [self.a_dim]),
                                    acts=([tf.nn.relu for _ in range(len(hidden_units_number))] + [None]),
                                    kp=self.dropout_keep_prob,
                                    use_bias=True)
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
    
    def train(self, kp=0.85):
        feed = {
            self.a: np.transpose(np.array(self.a_buffer), axes=[1, 0, 2]),
            self.r: np.transpose(np.array(self.r_buffer), axes=[1, 0]),
            self.s: np.transpose(np.array(self.s_buffer), axes=[1, 0, 2]),
            self.s_next: np.transpose(np.array(self.s_next_buffer), axes=[1, 0, 2]),
            self.dropout_keep_prob: kp
        }
        self.session.run([self.rl_train_op, self.sl_train_op], feed_dict=feed)
    
    def restore_buffer(self):
        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []
        self.s_next_buffer = []
    
    def save_current_state(self, s):
        self.s_buffer.append(s)
    
    def save_transition(self, a, r, s_next):
        self.a_buffer.append(a)
        self.r_buffer.append(r)
        self.s_next_buffer.append(s_next)
    
    def _trade(self, train=False, kp=1.0, prob=False):
        feed = {
            self.s: np.transpose(np.array(self.s_buffer), axes=[1, 0, 2]),
            self.dropout_keep_prob: kp
        }
        a_prob = self.session.run(self.a_out, feed_dict=feed)[:, -1, :]
        actions = []
        if train:
            for ap in a_prob:
                if prob:
                    ap = np.random.normal(loc=ap, scale=(1 - ap))
                    actions.append(np.exp(ap) / np.sum(np.exp(ap)))
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
    
    def load_model(self, model_path='./RPG_TF'):
        self.saver.restore(self.session, model_path + '/model')
    
    def save_model(self, model_path='./RPG_TF'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/model'
        self.saver.save(self.session, model_file)
    
    @staticmethod
    def create_new_model(asset_data,
                         c,
                         normalize_length,
                         batch_length,
                         train_length,
                         max_epoch,
                         learning_rate,
                         pass_threshold,
                         model_path):
        current_model_reward = -np.inf
        model = None
        while current_model_reward < pass_threshold:
            model = RPG_TF(s_dim=asset_data.shape[2],
                           b_dim=asset_data.shape[0],
                           a_dim=2,
                           learning_rate=learning_rate,
                           batch_size=batch_length,
                           normalize_length=normalize_length)
            model.init_model()
            model.restore_buffer()
            train_mean_r = []
            test_mean_r = []
            for e in range(max_epoch):
                test_reward = []
                test_actions = []
                train_reward = []
                previous_action = np.zeros(asset_data.shape[0])
                for t in range(model.normalize_length, train_length):
                    data = asset_data.iloc[:, t - model.normalize_length:t, :].values
                    state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
                    data = asset_data.iloc[:, t - model.normalize_length + 1:t + 1, :].values
                    next_state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
                    model.save_current_state(s=state)
                    action_ = model._trade(train=True, kp=1.0, prob=False)
                    r = asset_data[:, :, 'diff'].iloc[t].values * action_[:, 0] - c * np.abs(previous_action - action_[:, 0])
                    model.save_transition(a=action_, r=r, s_next=next_state)
                    previous_action = action_[:, 0]
                    train_reward.append(r)
                    if t % model.batch_size == 0:
                        model.train(kp=0.8)
                        model.restore_buffer()
                model.restore_buffer()
                print(e, 'train_reward', np.sum(np.mean(train_reward, axis=1)), np.mean(train_reward))
                train_mean_r.append(np.mean(train_reward))
                previous_action = np.zeros(asset_data.shape[0])
                for t in range(train_length, asset_data.shape[1]):
                    data = asset_data.iloc[:, t - model.normalize_length:t, :].values
                    state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
                    model.save_current_state(s=state)
                    action_ = model._trade(train=False, kp=1.0, prob=False)
                    r = asset_data[:, :, 'diff'].iloc[t].values * action_[:, 0] - c * np.abs(previous_action - action_[:, 0])
                    test_reward.append(r)
                    test_actions.append(action_)
                    previous_action = action_[:, 0]
                    if t % model.batch_size == 0:
                        model.restore_buffer()
                print(e, 'test_reward', np.sum(np.mean(test_reward, axis=1)), np.mean(test_reward))
                test_mean_r.append(np.mean(test_reward))
                model.restore_buffer()
                current_model_reward = np.sum(np.mean(test_reward, axis=1))
                if np.sum(np.mean(test_reward, axis=1)) > pass_threshold:
                    break
            model.restore_buffer()
        print('model created successfully, backtest reward:', current_model_reward)
        model.save_model(model_path)
        return model
    
    def back_test(self, asset_data, c, test_length):
        previous_action = np.zeros(asset_data.shape[0])
        test_reward = []
        test_actions = []
        for t in range(asset_data.shape[1] - test_length, asset_data.shape[1]):
            data = asset_data.iloc[:, t - self.normalize_length:t, :].values
            state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
            self.save_current_state(s=state)
            action_ = self._trade(train=False, kp=1.0, prob=False)
            r = asset_data[:, :, 'diff'].iloc[t].values * action_[:, 0] - c * np.abs(previous_action - action_[:, 0])
            test_reward.append(r)
            test_actions.append(action_)
            previous_action = action_[:, 0]
            if t % self.batch_size == 0:
                self.restore_buffer()
        self.restore_buffer()
        print('back test_reward', np.sum(np.mean(test_reward, axis=1)))
        return test_actions, test_reward
    
    def trade(self, asset_data):
        self.restore_buffer()
        for t in range(asset_data.shape[1] - self.batch_size, asset_data.shape[1]):
            data = asset_data.iloc[:, t - self.normalize_length + 1:t + 1, :].values
            state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
            self.save_current_state(s=state)
        action_ = self._trade(train=False, kp=1.0, prob=False)[:, 0]
        return action_/(np.sum(action_)+1e-10)
