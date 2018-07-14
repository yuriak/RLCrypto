# -*- coding:utf-8 -*-
import json
import importlib

LOG_FILE = './log/portfolio_log.csv'
BASE_CURRENCY = 'btc'
DEBUG_MODE = True
ORDER_TYPE = 'limit'
ACCOUNT_FILE = './config/account.json'


def init_config(config_path):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())
        system_config = config['system']
        model_config = config['models']
        trade_config = config['trade']
        train_config = config['train']
        test_config = config['test']
        data_config = config['data']
        global LOG_FILE
        LOG_FILE = system_config['log_file']
        global BASE_CURRENCY
        BASE_CURRENCY = trade_config['base_currency']
        global DEBUG_MODE
        DEBUG_MODE = trade_config['debug_mode']
        global PORTFOLIO_CONFIG
        PORTFOLIO_CONFIG = trade_config['portfolio_config']
        global ACCOUNT_FILE
        ACCOUNT_FILE = trade_config['account_file']
        global ORDER_TYPE
        ORDER_TYPE = trade_config['order_type']
        global PRICE_DISCOUNT
        PRICE_DISCOUNT = trade_config['price_discount']
        global AMOUNT_DISCOUNT
        AMOUNT_DISCOUNT = trade_config['amount_discount']
        global ORDER_WAIT_INTERVAL
        ORDER_WAIT_INTERVAL = trade_config['order_wait_interval']
        global TRACE_ORDER
        TRACE_ORDER = trade_config['trace_order']
        global TRADE_TRIGGER
        TRADE_TRIGGER = trade_config['trade_trigger']
        global MAX_ASSET_PERCENT
        MAX_ASSET_PERCENT = trade_config['max_asset_percent']
        global MAX_ORDER_WAITING_TIME
        MAX_ORDER_WAITING_TIME = trade_config['max_order_waiting_time']
        
        FEE = train_config['fee']
        global FEE
        NORMALIZE_LENGTH = train_config['normalize_length']
        global NORMALIZE_LENGTH
        BATCH_LENGTH = train_config['batch_length']
        global BATCH_LENGTH
        LEARNING_RATE = train_config['learning_rate']
        global LEARNING_RATE
        REWARD_THRESHOLD = train_config['reward_threshold']
        global REWARD_THRESHOLD
        MAX_TRAINING_EPOCH = train_config['max_training_epoch']
        global MAX_TRAINING_EPOCH
        TRAIN_LENGTH = train_config['train_length']
        global TRAIN_LENGTH
        
        TEST_LENGTH = test_config['test_length']
        global TEST_LENGTH
        
        TRAIN_BAR_COUNT = data_config['train_bar_count']
        global TRADE_BAR_COUNT
        TRADE_BAR_COUNT = data_config['trade_bar_count']
        global TRADE_BAR_COUNT
        TICK_INTERVAL = data_config['tick_interval']
        global TICK_INTERVAL
        
        MODEL_TYPE = trade_config['model_type']
        global MODEL_TYPE
        TRADER_MODEL = getattr(importlib.import_module("models.{0}".format(MODEL_TYPE)), MODEL_TYPE)
        global TRADER_MODEL
        HYPER_PARAMETERS = model_config[MODEL_TYPE]
        global HYPER_PARAMETERS
        MODEL_PATH = HYPER_PARAMETERS['model_path']
        global MODEL_PATH
