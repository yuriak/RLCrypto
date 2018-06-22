# -*- coding:utf-8 -*-
import sys
import json
from DataUtils import *
from TradingUtils import *
from model import *
from collections import OrderedDict
import shutil

LOG_FILE = 'portfolio_log.csv'
CONFIG_FILE = 'portfolio_config.json'
MODEL_PATH = './RPG_Portfolio1'

# portfolio selection
TRADING_TICK_INTERVAL = '30min'
PORTFOLIO_SELECTION_TICK_INTERVAL = '5min'
PORTFOLIO_SELECTION_BAR_COUNT = 500
BAR_COUNT = 2000
RISK_ASSET_NUMBER = 0
RISK_FREE_ASSET_NUMBER = 1
asset_symbols = []

# pre-processing parameters
MAX_PRE_PROCESSING_WINDOW = 20

# training hyper-parameters
REWARD_THRESHOLD = 0.5
FEE = 1e-4
NORMALIZE_LENGTH = 10
# total training length is BATCH_NUMBER*BATCH_SIZE
TRAIN_BATCH_SIZE = 40
TRAIN_BATCH_NUMBER = 40
MAX_TRAINING_EPOCH = 30
LEARNING_RATE = 1e-3
TEST_BATCH_NUMBER = 10
#  testing hyper-parameters
TEST_BATCH_SIZE = 50
TEST_BATCH_NUMBER = 10

# trading hyper-parameters
AMOUNT_DISCOUNT = 0.1
PRICE_DISCOUNT = -1e-4
DEBUG_MODE = True
BUY_ORDER_TYPE = 'limit'
SELL_ORDER_TYPE = 'market'


def select_coins(method='CAPM', risky_number=RISK_ASSET_NUMBER, risk_free_number=RISK_FREE_ASSET_NUMBER):
    symbols = lmap(lambda x: x['base-currency'], lfilter(lambda x: x['symbol-partition'] == 'innovation' and x['quote-currency'] == 'btc', get_symbols()['data']))
    print('fetching data')
    asset_data = klines(symbols, interval=PORTFOLIO_SELECTION_TICK_INTERVAL, count=PORTFOLIO_SELECTION_BAR_COUNT)
    print('building data')
    asset_data = OrderedDict(asset_data)
    asset_data = pd.Panel(asset_data)
    print(asset_data.shape)
    asset_data = asset_data.dropna(axis=1)
    asset_data.to_pickle('all_assets')
    market_index = asset_data[:, :, 'diff'].mean(axis=1)
    if method == 'CAPM':
        print('applying CAPM')
        capm = pd.DataFrame(lmap(lambda x: linreg(x=market_index.values, y=asset_data[x, :, 'diff'].values), asset_data.items), index=asset_data.items, columns=['alpha', 'beta'])
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


def create_new_model(asset_data, c=FEE, normalize_length=NORMALIZE_LENGTH, batch_size=TRAIN_BATCH_SIZE, batch_number=TRAIN_BATCH_NUMBER, max_epoch=MAX_TRAINING_EPOCH, learning_rate=LEARNING_RATE, pass_threshold=REWARD_THRESHOLD, model_path='./PG_Portfolio'):
    current_model_reward = -np.inf
    model = None
    while current_model_reward < pass_threshold:
        model = RPG_Crypto_portfolio(action_size=asset_data.shape[0] + 1, feature_number=asset_data.shape[2] * asset_data.shape[0], learning_rate=learning_rate)
        model.init_model()
        model.restore_buffer()
        train_mean_r = []
        test_mean_r = []
        for e in range(max_epoch):
            test_reward = []
            test_actions = []
            train_reward = []
            previous_action = np.zeros(asset_data.shape[0] + 1)
            for b in range(batch_number):
                for t in range(b * batch_size + normalize_length, (b + 1) * batch_size + normalize_length):
                    state = asset_data[:, t - normalize_length:t, :].values
                    state = state.reshape((state.shape[1], state.shape[0] * state.shape[2]))
                    state = z_score(state)[None, -1]
                    next_state = asset_data[:, t - normalize_length + 1:t + 1, :].values
                    next_state = next_state.reshape((next_state.shape[1], next_state.shape[0] * next_state.shape[2]))
                    next_state = z_score(next_state)[None, -1]
                    
                    model.save_current_state(s=state[0])
                    action = model.trade(state, train=True, drop=1.0)
                    r = np.sum(asset_data[:, :, 'diff'].iloc[t].values * action[:-1] - c * np.sum(np.abs(previous_action - action)))
                    model.save_transation(a=action, r=r, s_next=next_state[0])
                    previous_action = action
                    train_reward.append(r)
                loss = model.train(drop=1.0)
                model.restore_buffer()
            model.restore_buffer()
            print(e, 'train_reward', np.sum(train_reward), np.mean(train_reward))
            train_mean_r.append(np.mean(train_reward))
            previous_action = np.zeros(asset_data.shape[0] + 1)
            for t in range(batch_size * batch_number + normalize_length, asset_data.shape[1]):
                state = asset_data[:, t - normalize_length:t, :].values
                state = state.reshape((state.shape[1], state.shape[0] * state.shape[2]))
                state = z_score(state)[None, -1]
                model.save_current_state(s=state[0])
                action = model.trade(state, train=False)
                r = np.sum(asset_data[:, :, 'diff'].iloc[t].values * action[:-1] - c * np.sum(np.abs(previous_action - action)))
                previous_action = action
                if t % batch_size == 0:
                    model.restore_buffer()
                test_reward.append(r)
                test_actions.append(action)
            print(e, 'test_reward', np.sum(test_reward), np.mean(test_reward))
            test_mean_r.append(np.mean(test_reward))
            model.restore_buffer()
            current_model_reward = np.sum(test_reward)
            if np.sum(test_reward) > pass_threshold:
                break
        model.restore_buffer()
    print('model created successfully, backtest reward:', current_model_reward)
    model.save_model(model_path)
    return model


def backtest(asset_data, model, train_length=TRAIN_BATCH_NUMBER * TRAIN_BATCH_SIZE, batch_size=TEST_BATCH_SIZE, normalize_length=NORMALIZE_LENGTH, c=FEE):
    previous_action = np.zeros(asset_data.shape[0] + 1)
    test_reward = []
    test_actions = []
    for t in range(asset_data.shape[1] - train_length, asset_data.shape[1]):
        state = asset_data[:, t - normalize_length:t, :].values
        state = state.reshape((state.shape[1], state.shape[0] * state.shape[2]))
        state = z_score(state)[None, -1]
        model.save_current_state(s=state[0])
        action = model.trade(state, train=False, prob=False)
        r = np.sum(asset_data[:, :, 'diff'].iloc[t].values * action[:-1] - c * np.sum(np.abs(previous_action - action)))
        previous_action = action
        if t % batch_size == 0:
            model.restore_buffer()
        test_reward.append(r)
        test_actions.append(action)
    print('back test_reward', np.sum(test_reward))
    return np.sum(test_reward)


def real_trade(asset_data, assets, normalize_length=NORMALIZE_LENGTH, batch_size=TEST_BATCH_SIZE, debug=DEBUG_MODE, model_path=MODEL_PATH, log_path=LOG_FILE):
    model = RPG_Crypto_portfolio(action_size=asset_data.shape[0] + 1, feature_number=asset_data.shape[2] * asset_data.shape[0])
    model.load_model(model_path=model_path)
    backtest(asset_data, model=model)
    model.restore_buffer()
    for t in range(asset_data.shape[1] - batch_size, asset_data.shape[1]):
        state = asset_data[:, t - normalize_length + 1:t + 1, :].values
        state = state.reshape((state.shape[1], state.shape[0] * state.shape[2]))
        state = z_score(state)[None, -1]
        model.save_current_state(s=state[0])
        action = model.trade(state, train=False, prob=False)
    action = list(zip(assets, action[:-1]))
    action = sorted(action, key=lambda x: x[1])
    print('predict action', action)
    with open(log_path, 'a+') as f:
        f.write('{0},{1}\n'.format(int(asset_data[:, :, 'diff'].index[-1].timestamp()), str(action)))
        for k, a in action:
            print('sending order for asset', k)
            if a > 0:
                result = order_percent(a, symbol=k + 'btc', asset=k, order_type=BUY_ORDER_TYPE, debug=debug, price_discount=PRICE_DISCOUNT, amount_discount=AMOUNT_DISCOUNT)
                print(result)
            else:
                result = order_percent(a, symbol=k + 'btc', asset=k, order_type=SELL_ORDER_TYPE, debug=debug, price_discount=PRICE_DISCOUNT, amount_discount=AMOUNT_DISCOUNT)
                print(result)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'select':
        if os.path.exists(CONFIG_FILE):
            asset_symbols = json.loads(open(CONFIG_FILE, 'r+').read())
            for k in asset_symbols:
                order_percent(0, symbol=k + 'btc', asset=k, order_type='market', amount_discount=AMOUNT_DISCOUNT, debug=DEBUG_MODE)
            os.remove(CONFIG_FILE)
        asset_symbols = select_coins(risky_number=RISK_ASSET_NUMBER, risk_free_number=RISK_FREE_ASSET_NUMBER)
        if len(asset_symbols) == 0:
            print('no appropriate assets')
            sys.exit(1)
        with open(CONFIG_FILE, 'w+') as cf:
            cf.write(json.dumps(asset_symbols))
            print('writing selected assets...')
    elif mode == 'create_model':
        if not os.path.exists(CONFIG_FILE):
            print('config file not exist, please select coins first')
            sys.exit(1)
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
            print('deleting duplicated model...')
        with open(CONFIG_FILE, 'r+') as cf:
            asset_symbols = json.loads(cf.read())
        asset_data = klines(asset_symbols, interval=TRADING_TICK_INTERVAL, count=BAR_COUNT)
        asset_data = pre_process(asset_data, max_time_window=MAX_PRE_PROCESSING_WINDOW)
        print('start to create new model')
        create_new_model(asset_data=asset_data, model_path=MODEL_PATH)
    elif mode == 'trade':
        print('=' * 128)
        if not os.path.exists(CONFIG_FILE):
            print('config file not exist, please select coins first')
            sys.exit(1)
        if not os.path.exists(MODEL_PATH):
            print('model not exist, please create model first')
            sys.exit(1)
        with open(CONFIG_FILE, 'r+') as cf:
            asset_symbols = json.loads(cf.read())
        asset_data = klines(asset_symbols, interval=TRADING_TICK_INTERVAL, count=BAR_COUNT)
        asset_data = pre_process(asset_data, max_time_window=MAX_PRE_PROCESSING_WINDOW)
        # warning!!! set debug=False will lose all your money @_@
        real_trade(asset_data=asset_data, assets=asset_symbols, model_path=MODEL_PATH, log_path=LOG_FILE, debug=DEBUG_MODE)
    elif mode == 'sc':
        if os.path.exists(CONFIG_FILE):
            asset_symbols = json.loads(open(CONFIG_FILE, 'r+').read())
            for k in asset_symbols:
                order_percent(0, symbol=k + 'btc', asset=k, order_type=SELL_ORDER_TYPE, amount_discount=AMOUNT_DISCOUNT, debug=DEBUG_MODE)
            os.remove(CONFIG_FILE)
        asset_symbols = select_coins()
        if len(asset_symbols) == 0:
            print('no appropriate assets')
            sys.exit(1)
        with open(CONFIG_FILE, 'w+') as cf:
            cf.write(json.dumps(asset_symbols))
            print('writing selected assets...')
        if not os.path.exists(CONFIG_FILE):
            print('config file not exist, please select coins first')
            sys.exit(1)
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
            print('deleting duplicated model...')
        with open(CONFIG_FILE, 'r+') as cf:
            asset_symbols = json.loads(cf.read())
        asset_data = klines(asset_symbols, interval=TRADING_TICK_INTERVAL, count=BAR_COUNT)
        asset_data = pre_process(asset_data, max_time_window=MAX_PRE_PROCESSING_WINDOW)
        print('start to create new model')
        create_new_model(asset_data=asset_data, model_path=MODEL_PATH)
