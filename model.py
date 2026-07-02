# model.py
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from config import SPLIT_DATE, XGB_PARAMS, ALPHA_QUANTILE

def split_and_scale(X, y):
    X_train, X_test = X.loc[:SPLIT_DATE], X.loc[SPLIT_DATE:]
    y_train, y_test = y.loc[:SPLIT_DATE], y.loc[SPLIT_DATE:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def train_and_calibrate(X_train_scaled, y_train):
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train_scaled, y_train)
    
    # Calculate threshold filter on training probabilities
    raw_probs_in_sample = model.predict_proba(X_train_scaled)[:, 1]
    pct_cutoff_in_sample = np.quantile(raw_probs_in_sample, ALPHA_QUANTILE)
    
    return model, pct_cutoff_in_sample