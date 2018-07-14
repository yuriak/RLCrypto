# -*- coding:utf-8 -*-
import importlib
import json

log_file = "./log/portfolio_log.csv"
base_currency = 'btc'
debug_mode = True
portfolio_config = "./config/portfolio_config.json"
model_type = 'RPG_Torch'
model_path = './model_backup/RPG_Torch'
account_file = "./config/account.json"
order_type = 'limit'
price_discount = 1e-3
amount_discount = 0.05
trace_order = True
trade_time = [55]
max_asset_percent = 0.4
max_order_waiting_time = 120
fee = 1e-5
normalize_length = 10
batch_length = 64
learning_rate = 1e-3
reward_threshold = 0.3
max_training_epoch = 30
train_length = 1500
test_length = 400
trade_bar_count = 200
train_bar_count = 2000
tick_interval = '60min'
agent = getattr(importlib.import_module("models.{0}".format(model_type)), model_type)


# 我可能是个傻子。。。非要这么写？就为了用IDE的自动提示方便？
# 我可能确实是个傻子。。。
def init_config(config_path):
    with open(config_path, 'r') as f:
        global log_file
        global base_currency
        global debug_mode
        global portfolio_config
        global model_type
        global model_path
        global account_file
        global order_type
        global price_discount
        global amount_discount
        global trace_order
        global trade_time
        global max_asset_percent
        global max_order_waiting_time
        global fee
        global normalize_length
        global batch_length
        global learning_rate
        global reward_threshold
        global max_training_epoch
        global train_length
        global test_length
        global trade_bar_count
        global train_bar_count
        global tick_interval
        global agent
        config = json.loads(f.read())
        system_config = config['system']
        model_config = config['models']
        trade_config = config['trade']
        train_config = config['train']
        test_config = config['test']
        data_config = config['data']
        
        log_file = system_config['log_file']
        base_currency = trade_config['base_currency']
        debug_mode = trade_config['debug_mode']
        portfolio_config = trade_config['portfolio_config']
        model_type = trade_config['model_type']
        account_file = trade_config['account_file']
        order_type = trade_config['order_type']
        price_discount = trade_config['price_discount']
        amount_discount = trade_config['amount_discount']
        trace_order = trade_config['trace_order']
        trade_time = trade_config['trade_time']
        max_asset_percent = trade_config['max_asset_percent']
        max_order_waiting_time = trade_config['max_order_waiting_time']
        
        fee = train_config['fee']
        normalize_length = train_config['normalize_length']
        batch_length = train_config['batch_length']
        learning_rate = train_config['learning_rate']
        reward_threshold = train_config['reward_threshold']
        max_training_epoch = train_config['max_training_epoch']
        train_length = train_config['train_length']
        
        test_length = test_config['test_length']
        
        trade_bar_count = data_config['trade_bar_count']
        train_bar_count = data_config['train_bar_count']
        tick_interval = data_config['tick_interval']
        model_path = model_config[model_type]['model_path']
        
        agent = getattr(importlib.import_module("models.{0}".format(model_type)), model_type)
