# -*- coding:utf-8 -*-
import sys

import pandas as pd
import numpy as np
from model import PG_Crypto
from DataUtils import *
from TradingUtils import *
import datetime




def create_new_model(features):
    model = PG_Crypto(feature_number=features.shape[1])
    model.init_model()
    normalize_length = 10
    c = 1e-5
    for e in range(50):
        for t in range(normalize_length, 200 + normalize_length):
            s = features[t - normalize_length:t]
            s = np.expand_dims(((s - s.mean()) / s.std()).values[-1], axis=0)
            a = model.trade(s, train=True)
            if len(model.r_buffer) != 0:
                r = np.log(features.iloc[t]['diff']) * a - c * np.abs(model.a_buffer[-1] - a)
            else:
                r = np.log(features.iloc[t]['diff']) * a
            model.save_transation(a=a, s=s[0], r=r)
        loss = model.train()
        print(np.sum(model.r_buffer))
        if np.sum(model.r_buffer) > 0.2: break
        model.restore_buffer()
    test_reward, test_actions = back_test(model, features, length=1000, normalize_length=normalize_length, c=c)
    if np.sum(test_reward) > 1.5 * (np.max(np.log(features[features.shape[0] - 1000:]['diff']).cumsum()) - np.min(np.log(features[features.shape[0] - 1000:]['diff']).cumsum())):
        model.save_model()
        return model, np.sum(test_reward)
    else:
        return None, None


def back_test(model, features, length=1000, normalize_length=10, c=1e-5):
    test_reward = []
    test_actions = []
    for t in range(features.shape[0] - length, features.shape[0]):
        s = features[t - normalize_length:t]
        s = np.expand_dims(((s - s.mean()) / s.std()).values[-1], axis=0)
        a = model.trade(s, train=False)
        if len(model.r_buffer) != 0:
            r = np.log(features.iloc[t]['diff']) * a - c * np.abs(model.a_buffer[-1] - a)
        else:
            r = np.log(features.iloc[t]['diff']) * a - c
        test_reward.append(r)
        test_actions.append(a)
    return test_reward, test_actions


def trade(model, features, normalize_length=10):
    s = features[-normalize_length:]
    s = np.expand_dims(((s - s.mean()) / s.std()).values[-1], axis=0)
    a = model.trade(s, train=False)
    return a


if __name__ == '__main__':
    mode = sys.argv[1]
    with open('log.csv', 'a+') as f:
        if mode == 'create_model':
            k = get_kline('kanbtc', '15min', 2000)['data']
            s = pd.DataFrame(k)[::-1]
            s.index = pd.DatetimeIndex(s['id'].apply(lambda x: datetime.datetime.utcfromtimestamp(x) + datetime.timedelta(hours=8)))
            s = s.drop('id', axis=1)
            features = generate_tech_data(s, close_name='close', high_name='high', low_name='low', open_name='open')
            diff = (np.mean(s[['open', 'high', 'low', 'close']], axis=1) / np.mean(s[['open', 'high', 'low', 'close']], axis=1).shift(1)).fillna(1)
            diff.name = 'diff'
            features = features.join(diff)
            s_adj = s[features.index[0]:]
            model = None
            test_reward = None
            while model is None:
                model, test_reward = create_new_model(features)
            print('successfully build new model, test reward:', test_reward)
        elif mode == 'trade':
            k = get_kline('kanbtc', '15min', 2000)['data']
            s = pd.DataFrame(k)[::-1]
            s.index = pd.DatetimeIndex(s['id'].apply(lambda x: datetime.datetime.utcfromtimestamp(x) + datetime.timedelta(hours=8)))
            s = s.drop('id', axis=1)
            features = generate_tech_data(s, close_name='close', high_name='high', low_name='low', open_name='open')
            diff = (np.mean(s[['open', 'high', 'low', 'close']], axis=1) / np.mean(s[['open', 'high', 'low', 'close']], axis=1).shift(1)).fillna(1)
            diff.name = 'diff'
            features = features.join(diff)
            s_adj = s[features.index[0]:]
            model = None
            model = PG_Crypto(feature_number=features.shape[1])
            model.init_model()
            model.load_model()
            test_reward, test_arctions = back_test(model, features)
            print('back test reward', np.sum(test_reward))
            if np.sum(test_reward) > 1.2 * (np.max(np.log(features[features.shape[0] - 1000:]['diff']).cumsum()) - np.min(np.log(features[features.shape[0] - 1000:]['diff']).cumsum())):
                print('successfully pass backtest, ready to order!')
                action = trade(model, features)
                action = float(action)
                print('predict action', action)
                print('sending order', order_percent(action, debug=False))
                f.write('{0},{1}\n'.format(int(features.index[-1].timestamp()), action))
                print('order sent!')
            else:
                print('back test failed, please re-train model!')
                model = None
                while model is None:
                    model, test_reward = create_new_model(features)
                action = trade(model, features)
                action = float(action)
                print('predict action', action)
                print('sending order', order_percent(action, debug=False))
                f.write('{0},{1}\n'.format(int(features.index[-1].timestamp()), action))
                print('order sent!')
        else:
            print('fuck you')
