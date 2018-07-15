# -*- coding:utf-8 -*-
import sys
from utils.TradingUtils import *
from utils.config import *
from trader import *

CONFIG_PATH = './config/config.json'
if not os.path.exists(CONFIG_PATH):
    print("config file doesn't exist")
    sys.exit(1)
init_config(CONFIG_PATH)

init_account(account_file)
print(get_accounts())


class PortfolioManager(object):
    def __init__(self):
        self.portfolio = []
        self.asset_data = None
        self.agent = None
        self.trader = None
    
    def init_assets(self, assets_config):
        if not os.path.exists(assets_config):
            print('Portfolio config file does not exist, run PortfolioManager first')
            return
        with open(assets_config, 'r') as f:
            print('Load portfolio successfully')
            self.portfolio = json.loads(f.read())
    
    def init_data(self, bar_count):
        if len(self.portfolio) == 0:
            print('Load portfolio first')
            return
        asset_data = klines(self.portfolio, base_currency=base_currency, interval=tick_interval, count=bar_count)
        asset_data = default_pre_process(asset_data)
        self.asset_data = asset_data
    
    def init_trader(self):
        self.trader = Trader(assets=self.portfolio,
                             base_currency=base_currency,
                             max_asset_percent=max_asset_percent,
                             max_order_waiting_time=max_order_waiting_time,
                             price_discount=price_discount,
                             amount_discount=amount_discount,
                             order_type=order_type,
                             trace_order=trace_order,
                             debug_mode=debug_mode)
    
    def load_model(self):
        if len(self.portfolio) == 0 or self.asset_data is None:
            print('Init data first')
            return
        self.agent = agent(s_dim=self.asset_data.shape[-1],
                           b_dim=self.asset_data.shape[0],
                           a_dim=2,
                           learning_rate=learning_rate,
                           batch_length=batch_length,
                           normalize_length=normalize_length)
        self.agent.load_model(model_path=model_path)
    
    def build_model(self):
        if len(self.portfolio) == 0 or self.asset_data is None:
            print('Init data first')
            return
        self.agent = agent.create_new_model(asset_data=self.asset_data,
                                            c=fee,
                                            normalize_length=normalize_length,
                                            batch_length=batch_length,
                                            train_length=train_length,
                                            max_epoch=max_training_epoch,
                                            learning_rate=learning_rate,
                                            pass_threshold=reward_threshold,
                                            model_path=model_path)
    
    def back_test(self):
        if len(self.portfolio) == 0 or self.asset_data is None:
            print("Init data first")
            return
        self.agent.back_test(asset_data=self.asset_data, c=fee, test_length=test_length)
    
    def trade(self):
        print('=' * 100)
        if len(self.portfolio) == 0 or self.asset_data is None or self.agent is None:
            print('Init data and model')
            return
        actions = self.agent.trade(asset_data=self.asset_data)
        print('predict action for portfolio', list(zip(self.portfolio, actions)))
        self.trader.re_balance(actions=actions)
        print(datetime.datetime.now())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please input command')
        sys.exit(1)
    command = sys.argv[1]
    portfolio_manager = PortfolioManager()
    portfolio_manager.init_assets(assets_config=portfolio_config)
    if command == 'trade':
        last_trade_time = None
        portfolio_manager.init_data(trade_bar_count)
        portfolio_manager.load_model()
        print("Waiting to trade when triggered")
        while True:
            current_time = str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)
            if datetime.datetime.now().minute in trade_time and last_trade_time != current_time:
                print("Start to trade on {0}".format(datetime.datetime.now()))
                last_trade_time = current_time
                try:
                    portfolio_manager.init_data(trade_bar_count)
                except Exception as e:
                    portfolio_manager.init_data(trade_bar_count)
                portfolio_manager.trade()
    elif command == 'trade_now':
        try:
            portfolio_manager.init_data(trade_bar_count)
        except Exception:
            portfolio_manager.init_data(trade_bar_count)
        portfolio_manager.init_trader()
        portfolio_manager.load_model()
        portfolio_manager.trade()
    
    elif command == 'build_model':
        portfolio_manager.init_data(train_bar_count)
        portfolio_manager.build_model()
    elif command == 'backtest':
        portfolio_manager.init_data(train_bar_count)
        portfolio_manager.load_model()
        portfolio_manager.back_test()
    else:
        print('invalid command')
        # Donate XMR:   4AUY1FEpfGtYutRShAsmTMbVFmLoZdL92Gg6fQPYsN1P61mqrZpgnmsQKtYM8CkFpvDMJS6MuuKmncHhSpUtRyEqGcNUht2
        # :)
