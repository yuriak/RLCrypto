# -*- coding:utf-8 -*-
from threading import Thread
import time
from utils.HuobiServices import *
from utils.DataUtils import *

order_direction = {1: 'buy', -1: 'sell'}


class Trader(object):
    def __init__(self, assets,
                 base_currency,
                 max_asset_percent=0.5,
                 max_order_waiting_time=60,
                 price_discount=0.02,
                 amount_discount=0.05,
                 order_type='limit',
                 trace_order=True,
                 debug_mode=True):
        self.assets = assets
        self.base_currency = base_currency
        self.max_order_waiting_time = max_order_waiting_time
        self.price_discount = price_discount
        self.amount_discount = amount_discount
        self.max_asset_persent = max_asset_percent
        self.order_type = order_type
        self.trace_order = trace_order
        self.debug = debug_mode
        self.account = get_accounts()
        self.asset_info = self._get_asset_info()
        
        self.portfolio = None
    
    def init_portfolio(self):
        portfolio_balance = self._get_portfolio_balance()
        tickers = self._get_tickers()
        portfolio_value = self._get_portfolio_value(tickers, portfolio_balance)
        self.portfolio = pd.DataFrame({'value': portfolio_value, 'balance': portfolio_balance})
        return tickers
    
    def re_balance(self, actions):
        # batch cancel orders
        cancel_threads = []
        for a in self.assets:
            t = Thread(target=self._cancel_order, name='cancel_' + a, args=(a,), kwargs={'debug': self.debug})
            cancel_threads.append(t)
            time.sleep(1)
            t.start()
        [t.join() for t in cancel_threads]
        
        tickers = self.init_portfolio()
        target_weight = pd.Series(dict(zip(self.assets, actions)))
        target_weight = (target_weight / target_weight.sum()).fillna(0).clip(0, self.max_asset_persent)
        current_weight = (self.portfolio['value'] / self.portfolio['value'].sum())[self.assets]
        trade_weight = target_weight - current_weight
        trade_value = trade_weight * self.portfolio['value'].sum()
        trade_amount = (trade_value / tickers['close']).sort_values()
        sell_candidates = OrderedDict(trade_amount[trade_amount < 0])
        buy_candidates = OrderedDict(trade_amount[trade_amount > 0])
        
        # batch execute sell orders
        sell_threads = []
        for k, v in sell_candidates.items():
            amount, price, direction = self._generate_order(asset=k, trade_amount=v, tickers=tickers)
            t = Thread(target=self._execute_order,
                       name='sell_' + k,
                       args=(k, amount, price, direction),
                       kwargs={'trace_order': self.trace_order, 'debug': self.debug})
            sell_threads.append(t)
            time.sleep(1)
            t.start()
        [t.join() for t in sell_threads]
        
        # batch execute buy orders
        buy_threads = []
        for k, v in buy_candidates.items():
            amount, price, direction = self._generate_order(asset=k, trade_amount=v, tickers=tickers)
            t = Thread(target=self._execute_order,
                       name='sell_' + k,
                       args=(k, amount, price, direction),
                       kwargs={'trace_order': self.trace_order, 'debug': self.debug})
            buy_threads.append(t)
            time.sleep(1)
            t.start()
        [t.join() for t in buy_threads]
        return
    
    def _cancel_order(self, asset, debug=True):
        current_order_info = orders_list(symbol=(asset + self.base_currency), states='submitted')
        if current_order_info is None or 'data' not in current_order_info:
            return
        current_order_info = current_order_info['data']
        if len(current_order_info) > 0:
            for order in current_order_info:
                order_id = order['id']
                print('cancel previous order for {0}:'.format(asset + self.base_currency), order)
                if not debug:
                    cancel_order(order_id=order_id)
                else:
                    print('cancel order debugging')
    
    def _execute_order(self, asset, amount, price, direction, trace_order=False, debug=True):
        symbol = asset + self.base_currency
        print("*" * 25)
        print("send {0}-{1} order for {2}: "
              "on price: {3} with amount: {4}".format(self.order_type,
                                                      order_direction[direction],
                                                      symbol,
                                                      price,
                                                      direction * amount))
        if debug:
            print("debugging")
            return
        order = send_order(symbol=symbol,
                           source='api',
                           amount=amount,
                           _type=order_direction[direction] + '-' + self.order_type,
                           price=price if self.order_type == 'limit' else 0)
        print("order result for {0} {1}:".format(order_direction[direction], symbol), order)
        if order is None:
            return
        order_id = order['data']
        if order_id is None:
            return
        if not trace_order:
            return
        order_filled = False
        print("tracing order for {0} {1}".format(order_direction[direction], symbol))
        start_time = time.time()
        discounted_price = price
        while not order_filled:
            time.sleep(10)
            try:
                info = order_info(order_id)
            except Exception:
                info = order_info(order_id)
            if info is None or info['data'] is None:
                print('trace order failed for {0} {1}'.format(order_direction[direction], asset))
                return
            order_filled = (info['data']['state'] == 'filled')
            if (time.time() - start_time) > self.max_order_waiting_time:
                print("*" * 25)
                print("exceed pending time, use discount price for {0} {1}".format(order_direction[direction], symbol))
                try:
                    cancel_order(order_id)
                except Exception:
                    print('cancel order failed for {0} {1}'.format(order_direction[direction], symbol))
                    pass
                discounted_price = round(discounted_price * (1 + direction * self.price_discount), self.asset_info['pp'][asset])
                order = send_order(symbol=symbol,
                                   source='api',
                                   amount=amount,
                                   _type=order_direction[direction] + '-' + self.order_type,
                                   price=discounted_price)
                print("*" * 25)
                print("send {0}-{1} order for {2}: "
                      "on price: {3} with amount: {4}".format(self.order_type,
                                                              order_direction[direction],
                                                              symbol,
                                                              discounted_price,
                                                              direction * amount))
                start_time = time.time()
                print("order result for {0} {1}:".format(order_direction[direction], symbol), order)
                if order is None:
                    return
                order_id = order['data']
                if order_id is None:
                    return
        print("order full filled for {0} {1}".format(order_direction[direction], asset))
        return
    
    def _generate_order(self, asset, trade_amount, tickers):
        amount = (abs(trade_amount) * (1 - self.amount_discount))
        ap = self.asset_info['ap'][asset]
        amount = round(amount, ap) if ap > 0 else int(amount)
        direction = np.sign(trade_amount)
        
        pp = self.asset_info['pp'][asset]
        price = tickers['close'][asset] * (1 - direction * self.price_discount)
        price = round(price, pp) if pp > 0 else int(price)
        return amount, price, direction
    
    def _get_portfolio_balance(self):
        balance = lfilter(lambda x: x['currency'] in self.assets + [self.base_currency] and x['type'] == 'trade', get_balance()['data']['list'])
        portfolio_balance = dict(lmap(lambda x: (x['currency'], float(x['balance'])), balance))
        return pd.Series(portfolio_balance)
    
    def _get_tickers(self):
        tickers = lfilter(lambda x: x['symbol'] in lmap(lambda y: y + self.base_currency, self.assets), get_tickers()['data'])
        tickers = pd.DataFrame(tickers)
        tickers.index = tickers['symbol'].apply(lambda x: x.split(self.base_currency)[0])
        tickers.drop('symbol', axis=1, inplace=True)
        return tickers
    
    def _get_portfolio_value(self, tickers, balance):
        return (tickers['close'] * balance).fillna(balance[self.base_currency])
    
    def _get_asset_info(self):
        asset_info = lfilter(lambda x: x['base-currency'] in self.assets and x['quote-currency'] == self.base_currency, get_symbols()['data'])
        asset_info = dict(lmap(lambda x: (x['base-currency'], {'pp': int(x['price-precision']), 'ap': int(x['amount-precision'])}), asset_info))
        return pd.DataFrame(asset_info).T
