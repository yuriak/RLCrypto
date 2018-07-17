# -*- coding:utf-8 -*-
from models.Model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


class RNN(nn.Module):
    def __init__(self, s_dim, b_dim, out_dim=64, rnn_layers=1, dp=0.2):
        super(RNN, self).__init__()
        self.s_dim = s_dim
        self.b_dim = b_dim
        self.out_dim = out_dim
        self.rnn_layers = rnn_layers
        self.gru = nn.GRU(self.s_dim, self.out_dim, self.rnn_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dp)
        self.initial_hidden = torch.zeros(self.rnn_layers, self.b_dim, self.out_dim, dtype=torch.float32)
    
    def forward(self, state, hidden=None, train=False):
        if train:
            state = self.dropout(state)
        state, h = self.gru(state, hidden)
        return state, h.data


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, b_dim, dp=0.2):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.fc_actor_1 = nn.Linear(self.s_dim, 128)
        self.fc_actor_2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, self.a_dim)
        self.dropout = nn.Dropout(p=dp)
        self.relu = nn.ReLU()
    
    def forward(self, state, hidden=None, train=False):
        if train:
            state = self.dropout(state)
        a = self.relu(self.fc_actor_1(state))
        a = self.relu(self.fc_actor_2(a))
        a = self.relu(self.fc_out(a))
        if self.b_dim > 1:
            a = a / (a.sum(dim=0, keepdim=True) + 1e-10)
        a = a.clamp(0, 1)
        return a


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, dp=0.2):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim, 16)
        self.fc2 = nn.Linear(16 + a_dim, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dp)
    
    def forward(self, state, action, train=False):
        if train:
            state = self.dropout(state)
        q = self.relu(self.fc1(state))
        q = self.relu(self.fc2(torch.cat([q, action], dim=-1)))
        q = self.fc3(q)
        return q


class RDPG_Torch(Model):
    def __init__(self, s_dim, a_dim, b_dim, batch_length=64, learning_rate=1e-3, rnn_layers=1, normalize_length=10):
        super(RDPG_Torch, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.learning_rate = learning_rate
        self.batch_length = batch_length
        self.normalize_length = normalize_length
        self.tau = 0.001
        self.gamma = 0.99
        
        self.pointer = 0
        self.s_buffer = []
        self.a_buffer = []
        self.s_next_buffer = []
        self.r_buffer = []
        
        self.predict_train_hidden = None
        self.target_train_hidden = None
        
        self.trade_hidden = None
        
        self.rnn = RNN(s_dim=s_dim, b_dim=b_dim, out_dim=64, rnn_layers=rnn_layers)
        self.actor = Actor(s_dim=64, a_dim=a_dim, b_dim=b_dim)
        self.critic = Critic(s_dim=64, a_dim=a_dim)
        
        self.rnn_target = RNN(s_dim=s_dim, b_dim=b_dim, out_dim=64, rnn_layers=rnn_layers)
        self.actor_target = Actor(s_dim=64, a_dim=a_dim, b_dim=b_dim)
        self.critic_target = Critic(s_dim=64, a_dim=a_dim)
        
        self._hard_update(self.rnn_target, self.rnn)
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)
        
        self.actor_optim = optim.Adam(list(self.actor.parameters()) + list(self.rnn.parameters()), lr=self.learning_rate / 2)
        self.critic_optim = optim.Adam(list(self.critic.parameters()), lr=self.learning_rate)
        
        self.random_process = OrnsteinUhlenbeckProcess(a_dim=a_dim, b_dim=b_dim, n_steps_annealing=self.batch_length)
    
    def _trade(self, state, train=False, epsilon=0.):
        with torch.no_grad():
            s, self.trade_hidden = self.rnn(state[:, None, :], self.trade_hidden)
            a = self.actor(s, train=False)
        if train:
            a = (a + np.max([0, epsilon]) * self.random_process.sample()).clamp(0, 1)
            if self.b_dim > 1:
                a = a / (a.sum(dim=0, keepdim=True) + 1e-10)
            return a[:, 0, :]
        else:
            return a[:, 0, :]
    
    def _train(self, update_target=False):
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        state = torch.stack(self.s_buffer).t()
        state_next = torch.stack(self.s_next_buffer).t()
        reward = torch.stack(self.r_buffer).t()
        action_real = torch.stack(self.a_buffer).t()
        
        s, self.predict_train_hidden = self.rnn(state, self.predict_train_hidden)
        s_no_grad = s.detach()
        
        with torch.no_grad():
            s_next, self.target_train_hidden = self.rnn_target(state_next, self.target_train_hidden)
            a_next = self.actor_target(s_next)
            q_next = self.critic_target(state=s_next, action=a_next)
            q_target = reward + self.gamma * q_next
        
        self.actor.zero_grad()
        a_predict = self.actor(s)
        policy_loss = -self.critic(state=s, action=a_predict).mean()
        policy_loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.rnn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optim.step()
        
        self.critic.zero_grad()
        q_real = self.critic(state=s_no_grad, action=action_real)
        value_loss = F.smooth_l1_loss(q_real, q_target)
        value_loss.backward()
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.critic_optim.step()
        
        if update_target:
            self._soft_update(self.rnn_target, self.rnn, self.tau)
            self._soft_update(self.actor_target, self.actor, self.tau)
            self._soft_update(self.critic_target, self.critic, self.tau)
    
    def reset_model(self):
        self.s_buffer = []
        self.a_buffer = []
        self.s_next_buffer = []
        self.r_buffer = []
        self.trade_hidden = None
        self.predict_train_hidden = None
        self.target_train_hidden = None
        self.random_process.reset_states()
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
    
    def load_model(self, model_path='./RPG_Torch'):
        self.actor = torch.load(model_path + '/actor.pkl')
        self.critic = torch.load(model_path + '/critic.pkl')
        self.rnn = torch.load(model_path + '/rnn.pkl')
    
    def save_model(self, model_path='./RPG_Torch'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.actor, model_path + '/actor.pkl')
        torch.save(self.critic, model_path + '/model.pkl')
        torch.save(self.rnn, model_path + '/rnn.pkl')
    
    def train(self, asset_data, c, train_length, epoch=0, epsilon=1.):
        self.reset_model()
        previous_action = np.zeros(asset_data.shape[0])
        train_reward = []
        train_actions = []
        for t in range(self.normalize_length, train_length):
            data = asset_data.iloc[:, t - self.normalize_length:t, :].values
            state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
            state = torch.tensor(state)
            data = asset_data.iloc[:, t - self.normalize_length + 1:t + 1, :].values
            next_state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
            next_state = torch.tensor(next_state)
            action = self._trade(state, train=True, epsilon=epsilon)
            action_np = action.numpy().flatten()
            r = asset_data[:, :, 'diff'].iloc[t].values * action_np - c * np.abs(previous_action - action_np)
            self.save_transition(state=state, action=action, next_state=next_state, reward=r * 10)
            train_reward.append(r)
            train_actions.append(action_np)
            previous_action = action_np
            if t % self.batch_length // 2 == 0:
                self._train()
        self.reset_model()
        print(epoch, 'train_reward', np.sum(np.mean(train_reward, axis=1)), np.mean(train_reward))
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
            r = asset_data[:, :, 'diff'].iloc[t].values * action_np - c * np.abs(previous_action - action_np)
            test_reward.append(r)
            test_actions.append(action_np)
            previous_action = action_np
        self.reset_model()
        print(epoch, 'backtest reward', np.sum(np.mean(test_reward, axis=1)), np.mean(test_reward))
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
        return action_np
    
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
