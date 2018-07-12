# -*- coding:utf-8 -*-
import torch as torch
import torch.nn as nn
import torch.optim as optim
import os


class PolicyNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, b_dim, rnn_layers=1):
        super(PolicyNetwork, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.fc_1 = nn.Linear(self.s_dim, 128)
        self.gru = nn.GRU(128, 128, 1, batch_first=True)
        self.fc_s = nn.Linear(128, self.s_dim)
        self.fc_pg_1 = nn.Linear(128, 128)
        self.fc_pg_2 = nn.Linear(128, 64)
        self.fc_pg_out = nn.Linear(64, self.a_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.h = torch.zeros(1, self.b_dim, 128, dtype=torch.float32)
    
    def forward(self, x, hidden=None):
        x = self.relu(self.fc_1(x))
        if hidden is None:
            x, self.h = self.gru(x, self.h)
        else:
            x, self.h = self.gru(x, hidden)
        s_out = self.fc_s(x)
        pg_out = self.relu(self.fc_pg_1(x))
        pg_out = self.relu(self.fc_pg_2(pg_out))
        pg_out = self.softmax(self.fc_pg_out(pg_out))
        return pg_out, s_out, self.h.data
    
    def reset_hidden(self):
        self.h = torch.zeros(1, self.b_dim, 128, dtype=torch.float32)


class RecurrentPolicyGradient(object):
    def __init__(self, s_dim, a_dim, b_dim, batch_length=64, learing_rate=1e-3):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.batch_length = batch_length
        self.pointer = 0
        self.s_buffer = []
        self.a_buffer = []
        self.s_next_buffer = []
        self.r_buffer = []
        
        self.train_hidden = None
        self.trade_hidden = None
        self.policy = PolicyNetwork(s_dim=self.s_dim, a_dim=self.a_dim, b_dim=self.b_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learing_rate)
    
    def _trade(self, state, train=False):
        with torch.no_grad():
            a, _, self.trade_hidden = self.policy(state[:, None, :], self.trade_hidden)
        if train:
            return torch.multinomial(a[:, 0, :], 1)
        else:
            return a[:, 0, :].argmax(dim=1)
    
    def restore_buffer(self):
        self.s_buffer = []
        self.a_buffer = []
        self.s_next_buffer = []
        self.r_buffer = []
        self.policy.reset_hidden()
        self.pointer = 0
    
    def save_transition(self, state, action, reward, next_state):
        if self.pointer < self.batch_length:
            self.s_buffer.append(state)
            self.a_buffer.append(action)
            self.r_buffer.append(torch.tensor(reward[:, None], dtype=torch.float32))
            self.s_next_buffer.append(next_state)
            self.pointer += 1
        else:
            self.s_buffer.pop(0)
            self.a_buffer.pop(0)
            self.r_buffer.pop(0)
            self.s_next_buffer.pop(0)
            self.s_buffer.append(state)
            self.a_buffer.append(action)
            self.r_buffer.append(torch.tensor(reward[:, None], dtype=torch.float32))
            self.s_next_buffer.append(next_state)
    
    def _train(self):
        self.optimizer.zero_grad()
        s = torch.stack(self.s_buffer).t()
        s_next = torch.stack(self.s_next_buffer).t()
        r = torch.stack(self.r_buffer).t()
        a = torch.stack(self.a_buffer).t()
        a_hat, s_next_hat, self.train_hidden = self.policy(s, self.train_hidden)
        mse_loss = torch.nn.functional.mse_loss(s_next_hat, s_next)
        nll = -torch.log(a_hat.gather(2, a))
        pg_loss = (nll * r).mean()
        loss = mse_loss + pg_loss
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def load_model(self, model_path='./RecurrentPolicyGradient_PG'):
        self.policy = torch.load(model_path + '/model.pkl')
    
    def save_model(self, model_path='./RecurrentPolicyGradient_PG'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.policy, model_path + '/model.pkl')
