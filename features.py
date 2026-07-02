# features.py
import pandas as pd
import numpy as np

def generate_features(df):
    df = df.copy()
    
    # Technical Feature Indicators
    df['return_1_bar'] = np.log(df['Close'] / df['Close'].shift(1))
    df['return_4_bars'] = np.log(df['Close'] / df['Close'].shift(4))
    df['return_16_bars'] = np.log(df['Close'] / df['Close'].shift(16))
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['dist_from_ma20'] = df['Close'] - df['ma_20']
    df['volatility_20'] = df['Close'].rolling(window=20).std()
    df['hl_spread'] = df['High'] - df['Low']
    df['hour'] = df.index.hour
    df['is_ny_session'] = np.where((df['hour'] >= 13) & (df['hour'] <= 21), 1, 0)

    # Forward Lookahead Target Mapping
    df['future_close'] = df['Close'].shift(-4)
    df['target'] = np.where(df['future_close'] > df['Close'], 1, 0)
    df.dropna(inplace=True)

    feature_cols = ['return_1_bar', 'return_4_bars', 'return_16_bars', 'dist_from_ma20', 'volatility_20', 'hl_spread', 'is_ny_session']
    X = df[feature_cols]
    y = df['target']
    
    return df, X, y