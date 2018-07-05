# -*- coding:utf-8 -*-
import sys
from utils.DataUtils import *
from utils.TradingUtils import *

ACCOUNT_CONFIG_FILE = 'account.json'
LOG_FILE = 'portfolio_log.csv'
PORTFOLIO_CONFIG = 'portfolio_config.json'

ACCESS_KEY, SECRET_KEY = init_account(ACCOUNT_CONFIG_FILE)

# training hyper-parameters
REWARD_THRESHOLD = 0.3
FEE = 1e-5
NORMALIZE_LENGTH = 10
# total training length is BATCH_NUMBER*BATCH_SIZE
TRAIN_BATCH_SIZE = 64
TRAIN_LENGTH = 1500
MAX_TRAINING_EPOCH = 30
LEARNING_RATE = 1e-3

#  testing hyper-parameters
TEST_BATCH_SIZE = 64
TEST_LENGTH = 400

# trading hyper-parameters
AMOUNT_DISCOUNT = 0.02
PRICE_DISCOUNT = -5e-3
DEBUG_MODE = True
BUY_ORDER_TYPE = 'limit'
SELL_ORDER_TYPE = 'limit'
BASE_CURRENCY = 'btc'

class Trader(object):
    def __init__(self,log_file,portfolio_file):
        self.portfolio=[]
        self.model=None
    
    def build_model(self):
        pass
    
    def trade(self):
        pass