# -*- coding:utf-8 -*-
import sys
from models.Model import Model
from models.RecurrentPolicyGradient import RecurrentPolicyGradient
from models.PolicyGradient import PolicyGradient
from utils.DataUtils import default_pre_process
from utils.TradingUtils import *

MODEL_TYPE_CONFIG = './config/model_type.json'
MODEL_PATH = './model_backup/PolicyGradient'
ACCOUNT_CONFIG_FILE = './config/account.json'
LOG_FILE = './log/portfolio_log.csv'
PORTFOLIO_CONFIG = './config/portfolio_config.json'
TRADE_CONFIG = './config/trade_config.json'
init_account(ACCOUNT_CONFIG_FILE)
print(get_accounts())
with open(MODEL_TYPE_CONFIG, 'r') as f:
    model_type = json.loads(f.read())['model_type']
if model_type == 'RecurrentPolicyGradient':
    TRADER_MODEL = RecurrentPolicyGradient
else:
    TRADER_MODEL = PolicyGradient

# training hyper-parameters
REWARD_THRESHOLD = 0.3
FEE = 1e-5
NORMALIZE_LENGTH = 10

# total training length is BATCH_NUMBER*BATCH_SIZE
BATCH_SIZE = 64
TRAIN_LENGTH = 1500
MAX_TRAINING_EPOCH = 30
LEARNING_RATE = 1e-3

#  testing hyper-parameters
TEST_LENGTH = 400

# trading hyper-parameters
TRADING_TICK_INTERVAL = '60min'
BAR_COUNT = 2000
AMOUNT_DISCOUNT = 0.02
PRICE_DISCOUNT = -5e-3
BUY_ORDER_TYPE = 'limit'
SELL_ORDER_TYPE = 'limit'
with open(TRADE_CONFIG, 'r') as f:
    trade_config = json.loads(f.read())
    BASE_CURRENCY = trade_config['base_currency']
    DEBUG_MODE = trade_config['debug_mode']


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
    
    def init_data(self):
        if len(self.portfolio) == 0:
            print('Load portfolio first')
            return
        asset_data = klines(lmap(lambda x: x[0], self.portfolio), interval=TRADING_TICK_INTERVAL, count=BAR_COUNT)
        asset_data = default_pre_process(asset_data)
        self.asset_data = asset_data
    
    def load_model(self):
        if len(self.portfolio) == 0 or self.asset_data is None:
            print('Init data first')
            return
        self.model = TRADER_MODEL(s_dim=self.asset_data.shape[-1],
                                  a_dim=2,
                                  learning_rate=LEARNING_RATE,
                                  batch_size=BATCH_SIZE,
                                  normalize_length=NORMALIZE_LENGTH)
        self.model.load_model(model_path=MODEL_PATH)
    
    def build_model(self):
        if len(self.portfolio) == 0 or self.asset_data is None:
            print('Init data first')
            return
        self.model = TRADER_MODEL.create_new_model(asset_data_=self.asset_data,
                                                   c=FEE,
                                                   normalize_length=NORMALIZE_LENGTH,
                                                   batch_size=BATCH_SIZE,
                                                   train_length=TRAIN_LENGTH,
                                                   max_epoch=MAX_TRAINING_EPOCH,
                                                   learning_rate=LEARNING_RATE,
                                                   pass_threshold=REWARD_THRESHOLD,
                                                   model_path=MODEL_PATH)
    
    def trade(self):
        if len(self.portfolio) == 0 or self.asset_data is None or self.model is None:
            print('Init data and model')
            return
        actions = self.model.trade(asset_data_=self.asset_data)
        print('predict action for portfolio', zip(lmap(lambda x: x[0], self.portfolio), actions))
        for i in range(len(self.portfolio)):
            target_percent = actions[i]
            asset = self.portfolio[i][0]
            max_asset_percent = self.portfolio[i][1]
            re_balance(target_percent,
                       symbol=asset + BASE_CURRENCY,
                       asset=asset,
                       portfolio=lmap(lambda x: x[0], self.portfolio),
                       base_currency=BASE_CURRENCY,
                       order_type=BUY_ORDER_TYPE if target_percent > 0 else SELL_ORDER_TYPE,
                       price_discount=PRICE_DISCOUNT,
                       amount_discount=AMOUNT_DISCOUNT,
                       debug=DEBUG_MODE,
                       max_asset_percent=max_asset_percent)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please input command')
        sys.exit(1)
    command = sys.argv[1]
    trader = Trader()
    trader.init_portfolio(portfolio_config=PORTFOLIO_CONFIG)
    trader.init_data()
    if command == 'trade':
        trader.load_model()
        trader.trade()
    elif command == 'build_model':
        trader.build_model()
    else:
        print('invalid command')
        # 4AUY1FEpfGtYutRShAsmTMbVFmLoZdL92Gg6fQPYsN1P61mqrZpgnmsQKtYM8CkFpvDMJS6MuuKmncHhSpUtRyEqGcNUht2
