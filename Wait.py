# -*- coding:utf-8 -*-
from utils.DataUtils import *
from utils.TradingUtils import *

RISK_ASSET_NUMBER = 1
RISK_FREE_ASSET_NUMBER = 2
PORTFOLIO_SELECTION_TICK_INTERVAL = '5min'
PORTFOLIO_SELECTION_BAR_COUNT = 500
BASE_CURRENCY = 'btc'


class PortfolioManager(object):
    def __init__(self):
        pass
    
    def optimize_portfolio(self,method='CAPM', risky_number=RISK_ASSET_NUMBER, risk_free_number=RISK_FREE_ASSET_NUMBER):
        symbols = lmap(lambda x: x['base-currency'], lfilter(lambda x: x['symbol-partition'] == 'innovation' and x['quote-currency'] == BASE_CURRENCY, get_symbols()['data']))
        print('fetching data')
        data = klines(symbols, interval=PORTFOLIO_SELECTION_TICK_INTERVAL, count=PORTFOLIO_SELECTION_BAR_COUNT)
        print('building data')
        data = OrderedDict(data)
        data = pd.Panel(data)
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
