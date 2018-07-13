# -*- coding:utf-8 -*-
from abc import abstractmethod


class Model(object):
    def __init__(self):
        pass
    
    @abstractmethod
    def trade(self, asset_data):
        pass
    
    @abstractmethod
    def back_test(self, asset_data, c, test_length):
        pass
    
    @abstractmethod
    def load_model(self, model_path):
        pass
    
    @abstractmethod
    def save_model(self, model_path):
        pass
    
    @staticmethod
    @abstractmethod
    def create_new_model(asset_data,
                         c,
                         normalize_length,
                         batch_length,
                         train_length,
                         max_epoch,
                         learning_rate,
                         pass_threshold,
                         model_path):
        pass
