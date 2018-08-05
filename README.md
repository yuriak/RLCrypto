# RLCrypto

## Introduction
The trading framework for [Huobi Exchange](https://www.huobi.pro/)  
The Application of my research projects [RLQuant](https://github.com/yuriak/RLQuant) & [DLQuant](https://github.com/yuriak/DLQuant)

## Features
- Can be used to trade a portfolio
- Generate orders based on the actions of agents
- Trace orders and adjust prices if exceed the waiting threshold
- Multi-Thread order managing system
- Sets of configurable parameters in json
- Three models are integrated (Policy Gradient & Recurrent Policy Gradient & Direct RL)
- Models are implemented with two libraries (TF & Torch), (recommend to use Torch version, faster!)
- Setting trade time by yourself
- Find more from the code :\)

## Dependencies
- Python 3.5+
- numpy
- tensorflow
- pandas
- talib
- statsmodels
- PyTorch

## Models
- Multi-Task Recurrent Policy Gradient
    - Improved based on  [Recurrent Reinforcement Learning: A Hybrid Approach](https://arxiv.org/abs/1509.03044)
    - Integrate RNN to encode the temporal correlation
    - Predict the observation of the return rate of next step (supervised approach)
    - Feed the output of RNN into a vanilla Policy Gradient
    - Slow, but better than Vanilla PG
- Policy Gradient
    - The implementation of vanilla Policy Gradient [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
    - Simple, fast, stable
- Direct RL
    - Based on [Deep Direct Reinforcement Learning for Financial Signal Representation and Trading](http://ieeexplore.ieee.org/document/7407387/)
    - Directly optimize the expect return
    - Able to output continuous actions (weights)
    - Stable, can make good performance as RPG
## Usage
The interpretation of config file:
```
{
  "models": { //path of models
	"PG_TF": {
	  "model_path": "./model_backup/PG_TF"
	},
	"RPG_TF": {
	  "model_path": "./model_backup/RPG_TF"
	},
	"RPG_Torch": {
	  "model_path": "./model_backup/RPG_Torch"
	},
	"DRL_Torch": {
	  "model_path": "./model_backup/DRL_Torch"
	}
  },
  "data": {
	"trade_bar_count": 200,
	"train_bar_count": 2000,
	"tick_interval": "60min" //{"60min","30min","15min","5min"}
  },
  "train": {
	"fee": 1e-5,
	"normalize_length": 10, //longer: more stable, shorder: more sensitive  
	"batch_length": 64,
	"learning_rate": 1e-3,
	"reward_threshold": 0.3, // prevent overfitting, stop at this threshold
	"max_training_epoch": 30,
	"train_length": 1500
  },
  "test": {
	"test_length": 400
  },
  "trade": {
	"base_currency": "btc",
	"debug_mode": true,
	"portfolio_config": "./config/portfolio_config.json", //json file with a simple list of currency name: ["bat","soc","wicc"]
	"model_type": "DRL_Torch", //one of above
	"account_file": "./config/account.json", //{"ACCESS_KEY": "your access key","SECRET_KEY": "your secret key"}
	"order_type": "limit", // {"limit","market"} recommend to use limit order
	"price_discount": 5e-3, //the price will be adjusted if exceed order waiting time, 
	                            for buy order: new_price=price*(1+pd) i.e. use higher price to buy when the market lacks of liquidity. 
	                            for sell order: new_price=price*(1-pd) i.e. use lower price to sell
	"amount_discount": 0.09, // Avoid exceeding the maximum transaction amount caused by calculation errors  
	"trace_order": true, // recommend to use
	"trade_time": [ // The exact minute to trade, for example: on every hour's [15,30,45,59] to trade with 15min bars
	  55
	],
	"max_asset_percent": 0.4, //Max weight for a single asset. The holding percent of each currency in the portfolio cannot exceed this threshold.
	"max_order_waiting_time": 120 //When tracing orders, if the order cannot be full filled with in this value(second), 
	                                a new order will be generate with discounted price, until all actions are executed.
  },
  "system": {
	"log_file": "./log/portfolio_log.csv"
  }
}
```

To launch your AI trader:
1. Create a config file and modify it with your own parameters
```bash
copy ./config/config_template.json ./config/config.json
```
2. Create your account file with your access key and secret key
```bash
echo "{"ACCESS_KEY": "your access key","SECRET_KEY": "your secret key"}" >> ./config/account.json
```
3. Create your portfolio file with the asset you want to trade 
```bash
echo "["bat","soc", "wicc","edu"]" >> ./config/portfolio.json
```
4. Build your model
```bash
python3 portfolio_manager.py build_model
```
When it passes the backtest with the target return rate you set, it will automatically save the model parameters, then, you can use it to do live trading.

5. Live-Trading!!!

If you want to test if the system works correctly, you can choose to let it trade for once. 
```bash
python portfolio_manager.py trade_now
```
If all things are ready
- Set the ```debug_mode``` to ```false``` in your config file
- Make sure you have confidence in the market and agents  
- Prepare to launch you AI trader
```bash
python portfolio_manager.py trade
```
Or you can use the shell ```trade.sh```

6. Check the system periodically

## Risk Disclaimer (for Live-trading)
There is always risk of loss in trading. __All trading strategies are used at your own risk__  

The volumes of many cryptocurrency markets are still low. Market impact and slippage may badly affect the results during live trading.

## TODO
- Use the information of order book when managing orders
- Implement a security selection model based on maybe, fundamental data
- Maybe More

