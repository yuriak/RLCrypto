# -*- coding:utf-8 -*-
import sys
from models.RecurrentPolicyGradient import RecurrentPolicyGradient
from models.PolicyGradient import PolicyGradient
from utils.DataUtils import default_pre_process
from utils.TradingUtils import *

CONFIG_FILE = './config/config.json'
if not os.path.exists(CONFIG_FILE):
    print("config file doesn't exist")
    sys.exit(1)
with open(CONFIG_FILE, 'r') as f:
    config = json.loads(f.read())
    system_config = config['system']
    model_config = config['models']
    trade_config = config['trade']
    train_config = config['train']
    test_config = config['test']
    data_config = config['data']
    
    LOG_FILE = system_config['log_file']
    
    BASE_CURRENCY = trade_config['base_currency']
    DEBUG_MODE = trade_config['debug_mode']
    PORTFOLIO_CONFIG = trade_config['portfolio_config']
    ACCOUNT_FILE = trade_config['account_file']
    ORDER_TYPE = trade_config['order_type']
    PRICE_DISCOUNT = trade_config['price_discount']
    AMOUNT_DISCOUNT = trade_config['amount_discount']
    ORDER_WAIT_INTERVAL = trade_config['order_wait_interval']
    TRACE_ORDER = trade_config['trace_order']
    TRADE_TRIGGER = trade_config['trade_trigger']
    
    FEE = train_config['fee']
    NORMALIZE_LENGTH = train_config['normalize_length']
    BATCH_SIZE = train_config['batch_size']
    LEARNING_RATE = train_config['learning_rate']
    REWARD_THRESHOLD = train_config['reward_threshold']
    MAX_TRAINING_EPOCH = train_config['max_training_epoch']
    TRAIN_LENGTH = train_config['train_length']
    
    TEST_LENGTH = test_config['test_length']
    
    TRAIN_BAR_COUNT = data_config['train_bar_count']
    TRADE_BAR_COUNT = data_config['trade_bar_count']
    TICK_INTERVAL = data_config['tick_interval']
    
    MODEL_TYPE = trade_config['model_type']
    if MODEL_TYPE == 'RecurrentPolicyGradient':
        TRADER_MODEL = RecurrentPolicyGradient
    else:
        TRADER_MODEL = PolicyGradient
    HYPER_PARAMETERS = model_config[MODEL_TYPE]
    MODEL_PATH = HYPER_PARAMETERS['model_path']

init_account(ACCOUNT_FILE)
print(get_accounts())


class Trader(object):
    def __init__(self):
        self.portfolio = []
        self.asset_data = None
        self.model = None
    
    def init_portfolio(self, portfolio_config):
        if not os.path.exists(portfolio_config):
            print('Portfolio config file doesn\'t exist, run PortfolioManager first')
            return
        with open(portfolio_config, 'r') as f:
            print('Load portfolio successfully')
            self.portfolio = json.loads(f.read())
    
    def init_data(self, bar_count):
        if len(self.portfolio) == 0:
            print('Load portfolio first')
            return
        asset_data = klines(self.portfolio, base_currency=BASE_CURRENCY, interval=TICK_INTERVAL, count=bar_count)
        asset_data = default_pre_process(asset_data)
        self.asset_data = asset_data
    
    def load_model(self):
        if len(self.portfolio) == 0 or self.asset_data is None:
            print('Init data first')
            return
        self.model = TRADER_MODEL(s_dim=self.asset_data.shape[-1],
                                  a_dim=2,
                                  hidden_units_number=HYPER_PARAMETERS['hidden_units_number'],
                                  learning_rate=LEARNING_RATE,
                                  batch_size=BATCH_SIZE,
                                  normalize_length=NORMALIZE_LENGTH)
        self.model.load_model(model_path=MODEL_PATH)
    
    def build_model(self):
        if len(self.portfolio) == 0 or self.asset_data is None:
            print('Init data first')
            return
        self.model = TRADER_MODEL.create_new_model(asset_data=self.asset_data,
                                                   c=FEE,
                                                   hidden_units_number=HYPER_PARAMETERS['hidden_units_number'],
                                                   normalize_length=NORMALIZE_LENGTH,
                                                   batch_size=BATCH_SIZE,
                                                   train_length=TRAIN_LENGTH,
                                                   max_epoch=MAX_TRAINING_EPOCH,
                                                   learning_rate=LEARNING_RATE,
                                                   pass_threshold=REWARD_THRESHOLD,
                                                   model_path=MODEL_PATH)
    
    def trade(self):
        print('=' * 100)
        if len(self.portfolio) == 0 or self.asset_data is None or self.model is None:
            print('Init data and model')
            return
        actions = self.model.trade(asset_data=self.asset_data)
        print('predict action for portfolio', list(zip(self.portfolio, actions)))
        total = np.sum(actions)
        if total > 0:
            actions = actions / total
        actions = sorted(zip(self.portfolio, actions), key=lambda x: x[1])
        for asset, target_percent in actions:
            re_balance(target_percent,
                       symbol=asset + BASE_CURRENCY,
                       asset=asset,
                       portfolio=self.portfolio,
                       base_currency=BASE_CURRENCY,
                       order_type=ORDER_TYPE,
                       price_discount=PRICE_DISCOUNT,
                       amount_discount=AMOUNT_DISCOUNT,
                       debug=DEBUG_MODE,
                       max_asset_percent=target_percent,
                       wait_interval=ORDER_WAIT_INTERVAL,
                       trace_order=TRACE_ORDER)
        print(datetime.datetime.now())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please input command')
        sys.exit(1)
    command = sys.argv[1]
    trader = Trader()
    trader.init_portfolio(portfolio_config=PORTFOLIO_CONFIG)
    if command == 'trade':
        last_trade_hour = datetime.datetime.now().hour
        trader.init_data(TRADE_BAR_COUNT)
        trader.load_model()
        while True:
            if datetime.datetime.now().minute == TRADE_TRIGGER and last_trade_hour != datetime.datetime.now().hour:
                print("Start to trade on {0}".format(datetime.datetime.now()))
                last_trade_hour = datetime.datetime.now().hour
                trader.init_data(TRADE_BAR_COUNT)
                trader.trade()
    
    elif command == 'build_model':
        trader.init_data(TRAIN_BAR_COUNT)
        trader.build_model()
    else:
        print('invalid command')
        # Donate XMR:   4AUY1FEpfGtYutRShAsmTMbVFmLoZdL92Gg6fQPYsN1P61mqrZpgnmsQKtYM8CkFpvDMJS6MuuKmncHhSpUtRyEqGcNUht2
        # :)
