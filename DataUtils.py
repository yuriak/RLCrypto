# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import talib
import statsmodels.api as sm
from statsmodels import regression

lmap = lambda func, it: list(map(lambda x: func(x), it))
lfilter = lambda func, it: list(filter(lambda x: func(x), it))
z_score = lambda x: (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-5)


def generate_tech_data(stock, open_name, close_name, high_name, low_name, max_time_window=10):
    open_price = stock[open_name].values
    close_price = stock[close_name].values
    low_price = stock[low_name].values
    high_price = stock[high_name].values
    data = stock.copy()
    data['MOM'] = talib.MOM(close_price, timeperiod=max_time_window)
    data['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_price)
    data['HT_DCPHASE'] = talib.HT_DCPHASE(close_price)
    data['sine'], data['leadsine'] = talib.HT_SINE(close_price)
    data['inphase'], data['quadrature'] = talib.HT_PHASOR(close_price)
    data['ADXR'] = talib.ADXR(high_price, low_price, close_price, timeperiod=max_time_window)
    data['APO'] = talib.APO(close_price, fastperiod=max_time_window // 2, slowperiod=max_time_window)
    data['AROON_UP'], _ = talib.AROON(high_price, low_price, timeperiod=max_time_window)
    data['CCI'] = talib.CCI(high_price, low_price, close_price, timeperiod=max_time_window)
    data['PLUS_DI'] = talib.PLUS_DI(high_price, low_price, close_price, timeperiod=max_time_window)
    data['PPO'] = talib.PPO(close_price, fastperiod=max_time_window // 2, slowperiod=max_time_window)
    data['macd'], data['macd_sig'], data['macd_hist'] = talib.MACD(close_price, fastperiod=max_time_window // 2, slowperiod=max_time_window, signalperiod=max_time_window // 2)
    data['CMO'] = talib.CMO(close_price, timeperiod=max_time_window)
    data['ROCP'] = talib.ROCP(close_price, timeperiod=max_time_window)
    data['fastk'], data['fastd'] = talib.STOCHF(high_price, low_price, close_price)
    data['TRIX'] = talib.TRIX(close_price, timeperiod=max_time_window)
    data['ULTOSC'] = talib.ULTOSC(high_price, low_price, close_price, timeperiod1=max_time_window // 2, timeperiod2=max_time_window, timeperiod3=max_time_window * 2)
    data['WILLR'] = talib.WILLR(high_price, low_price, close_price, timeperiod=max_time_window)
    data['NATR'] = talib.NATR(high_price, low_price, close_price, timeperiod=max_time_window)
    data = data.drop([open_name, close_name, high_name, low_name], axis=1)
    data = data.dropna().astype(np.float32)
    return data





def find_cointegrated_pairs(dataframe):
    n = dataframe.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = dataframe.keys()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            stock1 = dataframe[keys[i]]
            stock2 = dataframe[keys[j]]
            result = sm.tsa.stattools.coint(stock1, stock2)
            pvalue = result[1]
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j], pvalue))
    return pvalue_matrix, pairs


def linreg(x, y):
    #     market, asset
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    x = x[:, 1]
    #     alpha beta
    return model.params[0], model.params[1]