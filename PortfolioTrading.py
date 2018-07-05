# -*- coding:utf-8 -*-
import shutil
import sys
from collections import OrderedDict

from utils.DataUtils import *

from models.experiment_models import *
from utils.TradingUtils import *

LOG_FILE = 'portfolio_log.csv'
CONFIG_FILE = 'portfolio_config.json'
MODEL_PATH = './RPG_Portfolio_Stable'

# portfolio selection
TRADING_TICK_INTERVAL = '60min'
PORTFOLIO_SELECTION_TICK_INTERVAL = '5min'
PORTFOLIO_SELECTION_BAR_COUNT = 500
BAR_COUNT = 2000
RISK_ASSET_NUMBER = 1
RISK_FREE_ASSET_NUMBER = 2
portfolio = []

# training hyper-parameters
REWARD_THRESHOLD = 0.45
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


def select_coins(method='CAPM', risky_number=RISK_ASSET_NUMBER, risk_free_number=RISK_FREE_ASSET_NUMBER):
    symbols = lmap(lambda x: x['base-currency'], lfilter(lambda x: x['symbol-partition'] == 'innovation' and x['quote-currency'] == BASE_CURRENCY, get_symbols()['data']))
    print('fetching data')
    data = klines(symbols, interval=PORTFOLIO_SELECTION_TICK_INTERVAL, count=PORTFOLIO_SELECTION_BAR_COUNT)
    print('building data')
    data = OrderedDict(data)
    data = pd.Panel(data)
    print(data.shape)
    data = data.dropna(axis=1)
    data.to_pickle('all_assets')
    market_index = data[:, :, 'diff'].mean(axis=1)
    if method == 'CAPM':
        print('applying CAPM')
        capm = pd.DataFrame(lmap(lambda x: linreg(x=market_index.values, y=data[x, :, 'diff'].values), data.items), index=data.items, columns=['alpha', 'beta'])
        high_risk = capm[(capm['alpha'] > 0) & (capm['beta'] > 1)].sort_values('alpha')
        low_risk = capm[(capm['alpha'] > 0) & (capm['beta'] < 1)].sort_values('alpha')
        print(len(high_risk), 'risky candidates')
        print(len(low_risk), 'risk-free candidates')
        candidate = []
        if risky_number > 0:
            candidate.extend(list(high_risk[-risky_number:].index))
        if risk_free_number > 0:
            candidate.extend(list(low_risk[-risk_free_number:].index))
        print(len(candidate))
        print(candidate)
        return candidate
    else:
        # not implemented
        return []


def create_new_model(asset_data_,
                     c=FEE,
                     normalize_length=NORMALIZE_LENGTH,
                     batch_size=TRAIN_BATCH_SIZE,
                     train_length=TRAIN_LENGTH,
                     max_epoch=MAX_TRAINING_EPOCH,
                     learning_rate=LEARNING_RATE,
                     pass_threshold=REWARD_THRESHOLD,
                     model_path=MODEL_PATH):
    current_model_reward = -np.inf
    model = None
    while current_model_reward < pass_threshold:
        model = RPG_Portfolio_Stable(action_size=2, feature_number=asset_data_.shape[2], learning_rate=learning_rate)
        model.init_model()
        model.restore_buffer()
        train_mean_r = []
        test_mean_r = []
        for e in range(max_epoch):
            test_reward = []
            test_actions = []
            train_reward = []
            previous_action = np.zeros(asset_data_.shape[0])
            for t in range(normalize_length, train_length):
                data = asset_data_[:, t - normalize_length:t, :].values
                state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
                data = asset_data_[:, t - normalize_length + 1:t + 1, :].values
                next_state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
                model.save_current_state(s=state)
                action_ = model.trade(train=True, kp=1.0, prob=False)
                r = asset_data_[:, :, 'diff'].iloc[t].values * action_[:, 0] - c * np.abs(previous_action - action_[:, 0])
                model.save_transation(a=action_, r=r, s_next=next_state)
                previous_action = action_[:, 0]
                train_reward.append(r)
                if t % batch_size == 0:
                    model.train(drop=0.8)
                    model.restore_buffer()
            model.restore_buffer()
            print(e, 'train_reward', np.sum(np.mean(train_reward, axis=1)), np.mean(train_reward))
            train_mean_r.append(np.mean(train_reward))
            previous_action = np.zeros(asset_data_.shape[0])
            for t in range(train_length, asset_data_.shape[0]):
                data = asset_data_[:, t - normalize_length:t, :].values
                state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
                model.save_current_state(s=state)
                action_ = model.trade(train=False, kp=1.0, prob=False)
                r = asset_data_[:, :, 'diff'].iloc[t].values * action_[:, 0] - c * np.abs(previous_action - action_[:, 0])
                test_reward.append(r)
                test_actions.append(action_)
                previous_action = action_[:, 0]
                if t % batch_size == 0:
                    model.restore_buffer()
            print(e, 'test_reward', np.sum(np.mean(test_reward, axis=1)), np.mean(test_reward))
            test_mean_r.append(np.mean(test_reward))
            model.restore_buffer()
            if np.sum(np.mean(test_reward, axis=1)) > pass_threshold:
                break
        model.restore_buffer()
    print('model created successfully, backtest reward:', current_model_reward)
    model.save_model(model_path)
    return model


def back_test(asset_data_, model, test_length=TEST_LENGTH, batch_size=TEST_BATCH_SIZE, normalize_length=NORMALIZE_LENGTH, c=FEE):
    previous_action = np.zeros(asset_data_.shape[0])
    test_reward = []
    test_actions = []
    for t in range(asset_data_.shape[1] - test_length, asset_data_.shape[1]):
        data = asset_data_[:, t - normalize_length:t, :].values
        state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
        model.save_current_state(s=state)
        action_ = model.trade(train=False, kp=1.0, prob=False)
        r = asset_data_[:, :, 'diff'].iloc[t].values * action_[:, 0] - c * np.abs(previous_action - action_[:, 0])
        test_reward.append(r)
        test_actions.append(action_)
        previous_action = action_[:, 0]
        if t % batch_size == 0:
            model.restore_buffer()
    model.restore_buffer()
    print('back test_reward', np.sum(np.mean(test_reward, axis=1)))
    return np.sum(np.mean(test_reward, axis=1))


def real_trade(asset_data_, portfolio_, normalize_length=NORMALIZE_LENGTH, batch_size=TEST_BATCH_SIZE, debug=DEBUG_MODE, model_path=MODEL_PATH):
    print('start to retrieve model from ', model_path)
    model = RPG_Portfolio_Stable(action_size=2, feature_number=asset_data_.shape[2])
    model.load_model(model_path=model_path)
    back_test(asset_data_, model=model)
    for t in range(asset_data_.shape[1] - batch_size, asset_data_.shape[0]):
        data = asset_data_[:, t - normalize_length + 1:t + 1, :].values
        state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
        model.save_current_state(s=state)
    action_ = model.trade(train=False, kp=1.0, prob=False)[:, 0]
    print('predict action', action_, 'for portfolio', portfolio_)
    for i in range(asset_data_.shape[0]):
        target_percent = action_[i]
        asset_ = portfolio_[i][0]
        max_asset_percent = portfolio_[i][1]
        if target_percent > 0:
            result = re_balance(target_percent,
                                symbol=asset_ + BASE_CURRENCY,
                                asset=asset_,
                                portfolio=lmap(lambda x: x[0], portfolio_),
                                base_currency=BASE_CURRENCY,
                                order_type=BUY_ORDER_TYPE,
                                price_discount=PRICE_DISCOUNT,
                                amount_discount=AMOUNT_DISCOUNT,
                                debug=debug,
                                max_asset_percent=max_asset_percent)
        else:
            result = re_balance(target_percent,
                                symbol=asset_ + BASE_CURRENCY,
                                asset=asset_,
                                portfolio=lmap(lambda x: x[0], portfolio_),
                                base_currency=BASE_CURRENCY,
                                order_type=SELL_ORDER_TYPE,
                                price_discount=PRICE_DISCOUNT,
                                amount_discount=AMOUNT_DISCOUNT,
                                debug=debug,
                                max_asset_percent=max_asset_percent)
        print(result)
    return dict(zip(lmap(lambda x: x[0], portfolio_), action_))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'select':
        print('=' * 100)
        if os.path.exists(CONFIG_FILE):
            portfolio = json.loads(open(CONFIG_FILE, 'r+').read())
            for asset, weight in portfolio:
                re_balance(0,
                           symbol=asset + BASE_CURRENCY,
                           asset=asset,
                           portfolio=lmap(lambda x: x[0], portfolio),
                           base_currency=BASE_CURRENCY,
                           order_type='market',
                           price_discount=0,
                           amount_discount=AMOUNT_DISCOUNT,
                           debug=DEBUG_MODE,
                           max_asset_percent=weight)
            os.remove(CONFIG_FILE)
        portfolio = select_coins(risky_number=RISK_ASSET_NUMBER, risk_free_number=RISK_FREE_ASSET_NUMBER)
        # TODO: applying markowitz portfolio optimization
        portfolio = lmap(lambda x: (x, 1.0 / len(portfolio)), portfolio)
        if len(portfolio) == 0:
            print('no appropriate assets')
            sys.exit(1)
        with open(CONFIG_FILE, 'w+') as cf:
            cf.write(json.dumps(portfolio))
            print('writing selected assets...')
    elif mode == 'create_model':
        print('=' * 100)
        if not os.path.exists(CONFIG_FILE):
            print('config file not exist, please select coins first')
            sys.exit(1)
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
            print('deleting duplicated model...')
        with open(CONFIG_FILE, 'r+') as cf:
            portfolio = json.loads(cf.read())
        asset_data = klines(lmap(lambda x: x[0], portfolio), interval=TRADING_TICK_INTERVAL, count=BAR_COUNT)
        asset_data = default_pre_process(asset_data)
        print('start to create new model')
        create_new_model(asset_data_=asset_data)
    elif mode == 'trade':
        print('=' * 100)
        if not os.path.exists(CONFIG_FILE):
            print('config file not exist, please select coins first')
            sys.exit(1)
        with open(CONFIG_FILE, 'r+') as cf:
            portfolio = json.loads(cf.read())
        if not os.path.exists(MODEL_PATH):
            print('model doesn\'t exist, please create model first')
            sys.exit(1)
        asset_data = klines(lmap(lambda x: x[0], portfolio), interval=TRADING_TICK_INTERVAL, count=BAR_COUNT)
        asset_data = default_pre_process(asset_data)
        # warning!!! set debug=False will lose all your money @_@
        print('start to trade:', portfolio)
        with open(LOG_FILE, 'a+') as f:
            trade_action=real_trade(asset_data_=asset_data,portfolio_=portfolio)
            f.write('{0},{1}\n'.format(int(asset_data[:, :, 'diff'].index[-1].timestamp()), str(trade_action)))
    elif mode == 'sc':
        if os.path.exists(CONFIG_FILE):
            portfolio = json.loads(open(CONFIG_FILE, 'r+').read())
            for asset, weight in portfolio:
                re_balance(0,
                           symbol=asset + BASE_CURRENCY,
                           asset=asset,
                           portfolio=lmap(lambda x: x[0], portfolio),
                           base_currency=BASE_CURRENCY,
                           order_type='market',
                           price_discount=0,
                           amount_discount=AMOUNT_DISCOUNT,
                           debug=DEBUG_MODE,
                           max_asset_percent=weight)
            os.remove(CONFIG_FILE)
        portfolio = select_coins(risky_number=RISK_ASSET_NUMBER, risk_free_number=RISK_FREE_ASSET_NUMBER)
        # TODO: applying markowitz portfolio optimization
        portfolio = lmap(lambda x: (x, 1.0 / len(portfolio)), portfolio)
        if len(portfolio) == 0:
            print('no appropriate assets')
            sys.exit(1)
        with open(CONFIG_FILE, 'w+') as cf:
            cf.write(json.dumps(portfolio))
            print('writing selected assets...')
        if not os.path.exists(CONFIG_FILE):
            print('config file not exist, please select coins first')
            sys.exit(1)
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
            print('deleting duplicated model...')
        with open(CONFIG_FILE, 'r+') as cf:
            portfolio = json.loads(cf.read())
        asset_data = klines(lmap(lambda x: x[0], portfolio), interval=TRADING_TICK_INTERVAL, count=BAR_COUNT)
        asset_data = default_pre_process(asset_data)
        print('start to create new model')
        create_new_model(asset_data_=asset_data)
