# RLCrypto

## Introduction
The trading framework for [Huobi Exchange](https://www.huobi.pro/)

## Features
- Can be used for trading a portfolio
- Automatically execute order based on the actions of agents
- Orders can be traced for filling
- Sets of configurable parameters in json
- Two models are integrated (Policy Gradient & Recurrent Policy Gradient)
- Periodically running
- Find more from the code :\)

## Dependencies
- Python 3.5+
- numpy
- tensorflow
- pandas
- talib
- statsmodels

## Models
- Recurrent Policy Gradient
    - Improved based on  [Recurrent Reinforcement Learning: A Hybrid Approach](https://arxiv.org/abs/1509.03044)
    - Integrate RNN to encode the temporal correlation
    - Predict the observation of next state (supervised approach)
    - Feed the output of RNN into a vanilla Policy Gradient
    - Slow, but better than Vanilla PG
- Policy Gradient
    - The implementation of vanilla Policy Gradient [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
    - Simple, fast, stable

## Usage
The interpretation of config file:
```json
{
  "models": {
//	hyper-parameters for PG  
	"PolicyGradient": { 
	  "hidden_units_number": [
		256,
		128,
		128,
		64
	  ],
	  "model_path": "./model_backup/PolicyGradient"
	},
//	hyper-parameters for RPG
	"RecurrentPolicyGradient": {
	  "hidden_units_number": [
		128,
		64
	  ],
	  "model_path": "./model_backup/RecurrentPolicyGradient"
	}
  },
  "data": {
	"trade_bar_count": 200, //data length for trading, should be longer than batch size
	"train_bar_count": 2000,//data length for trading, should be longer than train size
	"tick_interval": "60min"//tick interval, currently only support trade hourly
  },
  "train": {
	"fee": 1e-5, // fee for training
	"normalize_length": 10, //normalize data before feed into the network
	"batch_size": 64,
	"learning_rate": 1e-3,
	"reward_threshold": 0.3, // threshold for stopping training 
	"max_training_epoch": 30,
	"train_length": 1500
  },
  "test": {
	"test_length": 400
  },
  "trade": {
	"base_currency": "eth", // symbol: <basecurrency><asset>: (ethbat)
	"debug_mode": true,// order will not send to the exchange
	"portfolio_config": "./config/portfolio_config.json", // file content like ["eth","bat","xmr"]
	"model_type": "PolicyGradient",
	"account_file": "./config/account.json", //{"ACCESS_KEY": "","SECRET_KEY": ""}
	"order_type": "limit",
	"price_discount": -1e-3,//discount of limit order price, prevent not enough balance, negative for helping fill orders immediatly
	"amount_discount": 0.05, //prevent not enough balance
	"order_wait_interval": 0,//interval between orders
	"trace_order": true,
	"trade_trigger": 55,//the minute for trading in every hour
	"max_asset_percent": 0.5, // max asset weight for a single asset
	"max_order_waiting_time": 300 // exceed this time will trigger sending market order
  },
  "system": {
	"log_file": "./log/portfolio_log.csv"
  }
}
```

## Risk Disclaimer (for Live-trading)
There is always risk of loss in trading. __All trading strategies are used at your own risk__  

The volumes of many cryptocurrency markets are still low. Market impact and slippage may badly affect the results during live trading.

## TODO
- Implement DRPG for continuous action output
- Refactor with PyTorch
- Documents
- Maybe More

