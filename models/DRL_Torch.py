# -*- coding:utf-8 -*-
from models.Model import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class Actor(nn.Module):
    def __init__(self, s_dim, b_dim, rnn_layers=1, dp=0.2):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.b_dim = b_dim
        self.rnn_layers = rnn_layers
        self.gru = nn.GRU(self.s_dim, 128, self.rnn_layers, batch_first=True)
        self.fc_policy_1 = nn.Linear(128, 128)
        self.fc_policy_2 = nn.Linear(128, 64)
        self.fc_policy_out = nn.Linear(64, 1)
        self.fc_cash_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dp)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.initial_hidden = torch.zeros(self.rnn_layers, self.b_dim, 128, dtype=torch.float32)
    
    def forward(self, state, hidden=None, train=False):
        state, h = self.gru(state, hidden)
        if train:
            state = self.dropout(state)
        state = self.relu(self.fc_policy_1(state))
        state = self.relu(self.fc_policy_2(state))
        cash = self.sigmoid(self.fc_cash_out(state))
        action = self.sigmoid(self.fc_policy_out(state)).squeeze(-1).t()
        cash = cash.mean(dim=0)
        action = torch.cat(((1 - cash) * action, cash), dim=-1)
        action = action / (action.sum(dim=-1, keepdim=True) + 1e-10)
        return action, h.data


class DRL_Torch(Model):
    def __init__(self, s_dim, b_dim, a_dim=1, batch_length=64, learning_rate=1e-3, rnn_layers=1, normalize_length=10):
        self.s_dim = s_dim
        self.b_dim = b_dim
        self.batch_length = batch_length
        self.normalize_length = normalize_length
        self.pointer = 0
        self.s_buffer = []
        self.d_buffer = []
        
        self.train_hidden = None
        self.trade_hidden = None
        self.actor = Actor(s_dim=self.s_dim, b_dim=self.b_dim, rnn_layers=rnn_layers)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
    
    def _trade(self, state, train=False):
        with torch.no_grad():
            a, self.trade_hidden = self.actor(state[:, None, :], self.trade_hidden, train=False)
        return a
    
    def _train(self):
        self.optimizer.zero_grad()
        s = torch.stack(self.s_buffer).t()
        d = torch.stack(self.d_buffer)
        a_hat, self.train_hidden = self.actor(s, self.train_hidden, train=True)
        reward = -(a_hat[:, :-1] * d).mean()
        reward.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def reset_model(self):
        self.s_buffer = []
        self.d_buffer = []
        self.trade_hidden = None
        self.train_hidden = None
        self.pointer = 0
    
    def save_transition(self, state, reward):
        if self.pointer < self.batch_length:
            self.s_buffer.append(state)
            self.d_buffer.append(torch.tensor(reward, dtype=torch.float32))
            self.pointer += 1
        else:
            self.s_buffer.pop(0)
            self.d_buffer.pop(0)
            self.s_buffer.append(state)
            self.d_buffer.append(torch.tensor(reward, dtype=torch.float32))
    
    def load_model(self, model_path='./DRL_Torch'):
        self.actor = torch.load(model_path + '/model.pkl')
    
    def save_model(self, model_path='./DRL_Torch'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.actor, model_path + '/model.pkl')
    
    def train(self, asset_data, c, train_length, epoch=0):
        self.reset_model()
        previous_action = np.zeros(asset_data.shape[0])
        train_reward = []
        train_actions = []
        for t in range(self.normalize_length, train_length):
            data = asset_data.iloc[:, t - self.normalize_length:t, :].values
            state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
            state = torch.tensor(state)
            action = self._trade(state, train=True)
            action_np = action.numpy().flatten()
            r = asset_data[:, :, 'diff'].iloc[t].values * action_np[:-1] - c * np.abs(previous_action - action_np[:-1])
            self.save_transition(state=state, reward=asset_data[:, :, 'diff'].iloc[t].values)
            train_reward.append(r)
            train_actions.append(action_np)
            previous_action = action_np[:-1]
            if t % self.batch_length == 0:
                self._train()
        self.reset_model()
        print(epoch, 'train_reward', np.sum(np.sum(train_reward, axis=1)), np.mean(train_reward))
        return train_reward, train_actions
    
    def back_test(self, asset_data, c, test_length, epoch=0):
        self.reset_model()
        previous_action = np.zeros(asset_data.shape[0])
        test_reward = []
        test_actions = []
        for t in range(asset_data.shape[1] - test_length, asset_data.shape[1]):
            data = asset_data.iloc[:, t - self.normalize_length:t, :].values
            state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
            state = torch.tensor(state)
            action = self._trade(state=state, train=False)
            action_np = action.numpy().flatten()
            r = asset_data[:, :, 'diff'].iloc[t].values * action_np[:-1] - c * np.abs(previous_action - action_np[:-1])
            test_reward.append(r)
            test_actions.append(action_np)
            previous_action = action_np[:-1]
        self.reset_model()
        print(epoch, 'backtest reward', np.sum(np.sum(test_reward, axis=1)), np.mean(test_reward))
        return test_reward, test_actions
    
    def trade(self, asset_data):
        if self.trade_hidden is None:
            self.reset_model()
            action_np = np.zeros(asset_data.shape[0])
            for t in range(asset_data.shape[1] - self.batch_length, asset_data.shape[1]):
                data = asset_data.iloc[:, t - self.normalize_length + 1:t + 1, :].values
                state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
                state = torch.tensor(state)
                action = self._trade(state=state, train=False)
                action_np = action.numpy().flatten()
        else:
            data = asset_data.iloc[:, -self.normalize_length:, :].values
            state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
            state = torch.tensor(state)
            action = self._trade(state=state, train=False)
            action_np = action.numpy().flatten()
        return action_np[:-1]
    
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
            model = DRL_Torch(s_dim=asset_data.shape[2],
                              a_dim=2,
                              b_dim=asset_data.shape[0],
                              batch_length=batch_length,
                              learning_rate=learning_rate,
                              rnn_layers=1,
                              normalize_length=normalize_length)
            model.reset_model()
            for e in range(max_epoch):
                train_reward, train_actions = model.train(asset_data, c=c, train_length=train_length, epoch=e)
                test_actions, test_reward = model.back_test(asset_data, c=c, test_length=asset_data.shape[1] - train_length)
                current_model_reward = np.sum(np.mean(test_reward, axis=1))
                if current_model_reward > pass_threshold:
                    break
        print('model created successfully, backtest reward:', current_model_reward)
        model.save_model(model_path)
        return model
