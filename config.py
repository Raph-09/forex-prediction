# config.py
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/quant-data-archive/forex-historical/main/XAUUSD_M15.csv"
SPLIT_DATE = "2025-06-01"

XGB_PARAMS = {
    'n_estimators': 380, 'max_depth': 8, 'learning_rate': 0.02,
    'subsample': 0.85, 'colsample_bytree': 0.85, 'random_state': 42, 'eval_metric': 'logloss'
}

ALPHA_QUANTILE = 0.945
INITIAL_BALANCE = 100000.0
FIXED_LOT_SIZE = 5.5
SPREAD_AND_SLIPPAGE_PIPS = 1.5
FIXED_COST_PER_TRADE = (2.0 * 0.01 * FIXED_LOT_SIZE * 100) + (7.0 * FIXED_LOT_SIZE) + (SPREAD_AND_SLIPPAGE_PIPS * FIXED_LOT_SIZE * 10)

TAKE_PROFIT_DIST = 0.92
STOP_LOSS_DIST = 4.50
MAX_HOLD_BARS = 4
LOCKOUT_DURATION = 4
CIRCUIT_BREAKER_FLOOR = 0.91

# Advanced Sensitivity Parameter Space Matrix
SENSITIVITY_PARAM_GRID = {
    'lot_sizes': [4.5, 5.5, 6.5],
    'alpha_quantiles': [0.93, 0.945, 0.96]
}