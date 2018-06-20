# -*- coding:utf-8 -*-
from HuobiServices import *
import numpy as np
import pandas as pd

lmap = lambda func, it: list(map(lambda x: func(x), it))
lfilter = lambda func, it: list(filter(lambda x: func(x), it))


def kline(asset, interval='15min', count=500):
    s = get_kline('{0}btc'.format(asset), interval, count)
    if s is None: return None
    s = s['data']
    s = pd.DataFrame(s)[::-1]
    if s.shape[0] < count:
        return None
    s.index = pd.DatetimeIndex(s['id'].apply(lambda x: datetime.datetime.utcfromtimestamp(x) + datetime.timedelta(hours=8)))
    s = s.drop('id', axis=1)
    s['avg'] = (np.mean(s[['open', 'high', 'low', 'close']], axis=1))
    s['diff'] = np.log(s['avg'] / s['avg'].shift(1)).fillna(0)
    return s


def order_percent(target_percent, symbol='kanbtc', asset='kan', order_type='limit', price_discount=0, amount_discount=0.05, debug=True):
    balance = get_balance()
    asset_info = lfilter(lambda x: x['base-currency'] == asset and x['quote-currency'] == 'btc', get_symbols()['data'])
    current_order_info = orders_list(symbol=symbol, states='submitted')['data']
    
    amount_precision = asset_info[0]['amount-precision']
    price_precision = asset_info[0]['price-precision']
    btc_balance = list(filter(lambda x: x['currency'] == 'btc' and x['type'] == 'trade', balance['data']['list']))
    asset_balance = list(filter(lambda x: x['currency'] == asset and x['type'] == 'trade', balance['data']['list']))
    ticker = get_ticker(symbol)['tick']
    market_price = round(ticker['close'], price_precision)
    limit_buy_price = round(float(ticker['bid'][0]) * (1 - price_discount), price_precision)
    limit_sell_price = round(float(ticker['ask'][0]) * (1 + price_discount), price_precision)
    
    if len(current_order_info) > 0:
        for order in current_order_info:
            order_id = order['id']
            cancel_order(order_id=order_id)
    
    max_buy_amount = 0
    max_sell_amount = 0
    if len(btc_balance) > 0:
        max_buy_amount = float(btc_balance[0]['balance']) / float(market_price)
    if len(asset_balance) > 0:
        max_sell_amount = float(asset_balance[0]['balance'])
    available_balance = max_buy_amount + max_sell_amount
    holding = max_sell_amount / available_balance
    if target_percent > 0.9:
        target_percent = 1
    elif target_percent < 0.1:
        target_percent = 0
    trade_percent = target_percent - holding
    print('holding %', holding, 'hodling #', max_sell_amount)
    if trade_percent > 0.1:
        target_buy_amount = available_balance * trade_percent * (1 - amount_discount)
        if target_buy_amount > max_buy_amount:
            target_buy_amount = max_buy_amount
        target_buy_amount = round(target_buy_amount, amount_precision)
        if amount_precision == 0:
            target_buy_amount = int(target_buy_amount)
        if order_type == 'limit':
            print('send limit-buy order: buy #:', target_buy_amount, 'target holding %', target_percent * 100, '@price', limit_buy_price, 'on', symbol)
        else:
            print('send market-buy order: buy #:', target_buy_amount, 'target holding %', target_percent * 100, '@price', market_price, 'on', symbol)
        if not debug:
            if order_type == 'limit':
                order = send_order(symbol=symbol, source='api', amount=target_buy_amount, _type='buy-limit', price=limit_buy_price)
                print(order)
                return order['data']
            else:
                order = send_order(symbol=symbol, source='api', amount=target_buy_amount, _type='buy-market')
                print(order)
                return order['data']
        else:
            print('debugging')
            return 'debugging'
    elif trade_percent < -0.01:
        target_sell_amount = available_balance * np.abs(trade_percent) * (1 - amount_discount)
        if target_sell_amount > max_sell_amount:
            target_sell_amount = max_sell_amount
        target_sell_amount = round(target_sell_amount, amount_precision)
        if amount_precision == 0:
            target_sell_amount = int(target_sell_amount)
        if order_type == 'limit':
            print('send limit-sell order: sell #:', target_sell_amount, 'target holding %', target_percent * 100, '@price', limit_sell_price, 'on:', symbol)
        else:
            print('send market-sell order: sell #:', target_sell_amount, 'target holding %', target_percent * 100, '@price', market_price, 'on:', symbol)
        if not debug:
            if order_type == 'limit':
                order = send_order(symbol=symbol, source='api', amount=target_sell_amount, _type='sell-limit', price=limit_sell_price)
                print(order)
                return order['data']
            else:
                order = send_order(symbol=symbol, source='api', amount=target_sell_amount, _type='sell-market')
                print(order)
                return order['data']
        else:
            print('debugging')
            return 'debugging'
