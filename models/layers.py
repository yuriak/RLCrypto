# -*- coding:utf-8 -*-
import tensorflow as tf


def add_dense(inputs, units_numbers, acts, kp, use_bias=True):
    output = inputs
    for n, a in zip(units_numbers, acts):
        output = tf.layers.dense(output, n, activation=a, use_bias=use_bias)
        output = tf.nn.dropout(output, kp)
    return output


def add_GRU(units_number, activation=tf.nn.relu, kp=1.0):
    cell = tf.contrib.rnn.GRUCell(units_number, activation=activation)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=kp)
    return cell


def add_GRUs(units_numbers, acts, kp=1.0):
    cells = tf.contrib.rnn.MultiRNNCell(cells=[add_GRU(units_number=n, activation=a, kp=kp) for n, a in zip(units_numbers, acts)])
    return cells


def add_LSTM(units_number, activation=tf.nn.relu, kp=1.0):
    cell = tf.contrib.rnn.LSTMCell(units_number, activation=activation)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=kp)
    return cell


def add_LSTMs(units_numbers, acts, kp=1.0):
    cells = tf.contrib.rnn.MultiRNNCell(cells=[add_LSTM(units_number=n, activation=a, kp=kp) for n, a in zip(units_numbers, acts)])
    return cells
