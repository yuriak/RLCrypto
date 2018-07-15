# -*- coding:utf-8 -*-
from models.Model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, b_dim, rnn_layers=1, dp=0.2):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.rnn_layers = rnn_layers
        self.gru = nn.GRU(self.s_dim, 128, self.rnn_layers, batch_first=True)
        self.fc_actor_1 = nn.Linear(128, 128)
        self.fc_actor_2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dp)
        self.initial_hidden = torch.zeros(self.rnn_layers, self.b_dim, 128, dtype=torch.float32)
    
    def forward(self, state, hidden=None, train=False):
        state, h = self.gru(state, hidden)
        if train:
            state = self.dropout(state)
        actor_out = self.relu(self.fc_actor_1(state))
        actor_out = self.relu(self.fc_actor_2(actor_out))
        if train:
            actor_out = self.dropout(actor_out)
        actor_out = actor_out / (actor_out.sum(dim=0, keepdim=True) + 1e-10)
        return actor_out, h.data


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, dp=0.2):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim, 16)
        self.fc2 = nn.Linear(16 + 1, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dp)
    
    def forward(self, state, action, train=False):
        if train:
            state = self.dropout(state)
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, action], dim=-1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


class RDPG_Torch(Model):
    def __init__(self, s_dim, a_dim, b_dim, batch_length=64, learning_rate=1e-3, rnn_layers=1, normalize_length=10):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.batch_length = batch_length
        self.normalize_length = normalize_length
        self.tau = 0.001
        self.discount = 0.99
        self.depsilon = 1.0 / 30
        self.epsilon = 1.0
        
        self.pointer = 0
        self.s_buffer = []
        self.a_buffer = []
        self.s_next_buffer = []
        self.r_buffer = []
        
        self.train_hidden = None
        self.trade_hidden = None
        
        self.actor = Actor(s_dim=s_dim, a_dim=a_dim, b_dim=b_dim, rnn_layers=rnn_layers)
        self.actor_target = Actor(s_dim=s_dim, a_dim=a_dim, b_dim=b_dim, rnn_layers=rnn_layers)
        
        self.critic = Critic(s_dim=s_dim, a_dim=a_dim)
        self.critic_target = Critic(s_dim=s_dim, a_dim=a_dim)
        
        self.random_process = OrnsteinUhlenbeckProcess(a_dim=a_dim, b_dim=b_dim, n_steps_annealing=self.batch_length)
    
    def _trade(self, state, train=False):
        with torch.no_grad():
            a, self.trade_hidden = self.actor(state[:, None, :], self.trade_hidden, train=False)
        if train:
            a = a * self.random_process.sample()
            return a / (a.sum(dim=0, keepdim=True) + 1e-10)
        else:
            return a
    
    def load_model(self, model_path):
        pass
    
    def save_model(self, model_path):
        pass
    
    def back_test(self, asset_data, c, test_length):
        pass
    
    def trade(self, asset_data):
        pass
    
    @staticmethod
    def create_new_model(asset_data, c, normalize_length, batch_length, train_length, max_epoch, learning_rate, pass_threshold, model_path):
        pass
    
    @staticmethod
    def _hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    @staticmethod
    def _soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )


# [reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py

class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, a_dim, b_dim, mu, sigma, sigma_min, n_steps_annealing):
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.mu = torch.ones(self.b_dim, 1, self.a_dim) * mu
        self.sigma = torch.ones(self.b_dim, 1, self.a_dim) * sigma
        self.n_steps = 0
        
        if sigma_min is not None:
            self.m = (self.sigma - sigma_min) / n_steps_annealing
            self.c = self.sigma
            self.sigma_min = sigma_min
        else:
            self.m = torch.zeros(self.b_dim, 1, self.a_dim)
            self.c = self.sigma
            self.sigma_min = self.sigma
    
    @property
    def current_sigma(self):
        sigma = torch.max(self.sigma_min, self.m * self.n_steps + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, b_dim, a_dim, theta=0.15, mu=0., sigma=0.2, dt=1e-2, x0=None, sigma_min=None, n_steps_annealing=100):
        super(OrnsteinUhlenbeckProcess, self).__init__(a_dim=a_dim, b_dim=b_dim, mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.b_dim = b_dim
        self.a_dim = a_dim
        self.theta = torch.tensor(theta)
        self.mu = torch.ones(b_dim, 1, a_dim) * mu
        self.dt = torch.tensor(dt)
        self.x0 = x0
        self.reset_states()
    
    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * torch.sqrt(self.dt) * torch.randn(self.b_dim, 1, self.a_dim)
        self.x_prev = x
        self.n_steps += 1
        return x
    
    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else torch.zeros(self.b_dim, 1, self.a_dim)
