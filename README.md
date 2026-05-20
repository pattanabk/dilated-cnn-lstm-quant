# Gold Volatility Quant Model

Deep learning quantitative research project for gold volatility forecasting using:

- Dilated CNN
- LSTM
- Multi-market macro features
- Volatility regime prediction
- 1H timeframe market data
- PyTorch

Includes:
- Full model version
- CPU-optimized lightweight version

## Files

- dilated_cnn_lstm.py
- dilated_cnn_lstm_small.py
- requirements.txt

## Features

- Realized volatility forecasting
- Regime classification
- Multi-asset macro features
- OHLC forecasting
- Quant research architecture

## Sample Results

### V31.2 FAST CPU RESULTS

VOL MAE         : 0.003899
VOL RMSE        : 0.005953
REGIME ACCURACY : 0.377193

### Example Forecast

LAST CLOSE       : 4743.50
EXPECTED MOVE    : 34.30
EXPECTED VOL     : 0.007232
NEXT REGIME      : EXPANSION

## Model Architecture

- Dilated CNN feature extraction
- LSTM temporal modeling
- Multi-task learning
- Volatility forecasting
- Regime classification
- Cross-market macro features

## Future Improvements

- Transformer architectures
- Real-time websocket feeds
- Cross-exchange arbitrage integration
- Reinforcement learning execution
- Orderbook microstructure features

## Disclaimer

Research/educational purposes only.
Not financial advice.
