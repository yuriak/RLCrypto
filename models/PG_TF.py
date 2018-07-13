# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from models.Model import Model
from models.layers import *


class PG_TF(Model):
    def __init__(self, s_dim, b_dim, a_dim=2, hidden_units_number=[256, 128, 128, 64], learning_rate=0.001, batch_size=64, normalize_length=10):
        super(PG_TF, self).__init__()
        tf.reset_default_graph()
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, None, s_dim], name='s')
        self.a = tf.placeholder(dtype=tf.int32, shape=[None, None, a_dim], name='a')
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, None], name='r')
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.b_dim = b_dim
        self.batch_size = batch_size
        self.normalize_length = normalize_length
        
        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        with tf.variable_scope('policy', initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            self.a_prob = add_dense(inputs=self.s,
                                    units_numbers=hidden_units_number + [self.a_dim],
                                    acts=[tf.nn.relu for _ in range(len(hidden_units_number))] + [None],
                                    kp=self.dropout_keep_prob,
                                    use_bias=True)
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
    
    def train(self, kp=0.85):
        random_index = np.arange(len(self.s_buffer))
        np.random.shuffle(random_index)
        feed = {
            self.a: np.transpose(np.array(self.a_buffer)[random_index], axes=[1, 0, 2]),
            self.r: np.transpose(np.array(self.r_buffer)[random_index], axes=[1, 0]),
            self.s: np.transpose(np.array(self.s_buffer)[random_index], axes=[1, 0, 2]),
            self.dropout_keep_prob: kp
        }
        self.session.run(self.train_op, feed_dict=feed)
    
    def restore_buffer(self):
        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []
    
    def save_transation(self, s, a, r):
        self.a_buffer.append(a)
        self.r_buffer.append(r)
        self.s_buffer.append(s)
    
    def _trade(self, s, train=False, kp=1.0, prob=False):
        feed = {
            self.s: s[:, None, :],
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
    
    def load_model(self, model_path='./PG_TF'):
        self.saver.restore(self.session, model_path + '/model')
    
    def save_model(self, model_path='./PG_TF'):
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
            model = PG_TF(s_dim=asset_data.shape[2],
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
                    data = asset_data[:, t - model.normalize_length:t, :].values
                    state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
                    action = model._trade(state, train=True, prob=False, kp=1.0)
                    r = asset_data[:, :, 'diff'].iloc[t].values * action[:, 0] - c * np.abs(previous_action - action[:, 0])
                    model.save_transation(a=action, s=state, r=r)
                    previous_action = action[:, 0]
                    train_reward.append(r)
                    if t % model.batch_size == 0:
                        model.train(kp=0.8)
                        model.restore_buffer()
                model.restore_buffer()
                print(e, 'train_reward', np.sum(np.mean(train_reward, axis=1)), np.mean(train_reward))
                train_mean_r.append(np.mean(train_reward))
                previous_action = np.zeros(asset_data.shape[0])
                for t in range(train_length, asset_data.shape[1]):
                    data = asset_data[:, t - model.normalize_length:t, :].values
                    state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
                    action = model._trade(state, train=True, prob=False, kp=1.0)
                    r = asset_data[:, :, 'diff'].iloc[t].values * action[:, 0] - c * np.abs(previous_action - action[:, 0])
                    test_reward.append(r)
                    test_actions.append(action)
                    previous_action = action[:, 0]
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
        test_reward = []
        test_actions = []
        previous_action = np.zeros(asset_data.shape[0])
        for t in range(asset_data.shape[1] - test_length, asset_data.shape[1]):
            data = asset_data[:, t - self.normalize_length:t, :].values
            state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
            action = self._trade(state, train=False, prob=False)
            r = asset_data[:, :, 'diff'].iloc[t].values * action[:, 0] - c * np.abs(previous_action - action[:, 0])
            test_reward.append(r)
            test_actions.append(action)
            previous_action = action[:, 0]
        self.restore_buffer()
        print('back test_reward', np.sum(np.mean(test_reward, axis=1)))
        return test_actions, test_reward
    
    def trade(self, asset_data):
        self.restore_buffer()
        data = asset_data[:, -self.normalize_length:, :].values
        state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
        action_ = self._trade(state, train=False, prob=False, kp=1.0)[:, 0]
        return action_
