# backtester.py
import numpy as np
import pandas as pd
from config import (INITIAL_BALANCE, FIXED_LOT_SIZE, FIXED_COST_PER_TRADE, 
                    MAX_HOLD_BARS, LOCKOUT_DURATION, CIRCUIT_BREAKER_FLOOR)

def run_backtest(prices_df, base_signals, is_oos=False, dynamic_lot=None, dynamic_alpha_cutoff=None):
    price_swings = prices_df['Close'].diff().fillna(0)
    rolling_vol = price_swings.rolling(14).std().fillna(price_swings.std()).values

    close_prices = prices_df['Close'].values
    high_prices = prices_df['High'].values
    low_prices = prices_df['Low'].values
    timestamps = prices_df.index

    balance = INITIAL_BALANCE
    trade_log = []          # Net dollar returns per trade
    equity_curve = []       # Balance state at every step
    trade_durations = []    # Tracking bar holds for micro analysis
    
    lot_size = dynamic_lot if dynamic_lot is not None else FIXED_LOT_SIZE
    cost_per_trade = (2.0 * 0.01 * lot_size * 100) + (7.0 * lot_size) + (1.5 * lot_size * 10)

    in_position = False
    entry_price, current_sl, current_tp, bars_held, lockout_timer = 0, 0, 0, 0, 0

    for i in range(len(base_signals)):
        current_close = close_prices[i]
        current_high = high_prices[i]
        current_low = low_prices[i]
        vol_scale = rolling_vol[i] if rolling_vol[i] > 0 else 1.0

        if is_oos and balance < (INITIAL_BALANCE * CIRCUIT_BREAKER_FLOOR):
            if in_position:
                net_pnl = ((current_close - entry_price) * lot_size * 100) - cost_per_trade
                balance += net_pnl; trade_log.append(net_pnl); trade_durations.append(bars_held)
                in_position = False
            equity_curve.append(balance)
            continue

        if lockout_timer > 0:
            lockout_timer -= 1

        if in_position:
            bars_held += 1
            if bars_held >= 2:
                current_sl = max(current_sl, entry_price + 0.38)

            hit_tp = current_high >= current_tp
            hit_sl = current_low <= current_sl

            if hit_tp and hit_sl:
                net_pnl = ((current_sl - entry_price) * lot_size * 100) - cost_per_trade
                balance += net_pnl; trade_log.append(net_pnl); trade_durations.append(bars_held); in_position = False; lockout_timer = LOCKOUT_DURATION
            elif hit_tp:
                net_pnl = ((current_tp - entry_price) * lot_size * 100) - cost_per_trade
                balance += net_pnl; trade_log.append(net_pnl); trade_durations.append(bars_held); in_position = False; lockout_timer = LOCKOUT_DURATION
            elif hit_sl:
                net_pnl = ((current_sl - entry_price) * lot_size * 100) - cost_per_trade
                balance += net_pnl; trade_log.append(net_pnl); trade_durations.append(bars_held); in_position = False; lockout_timer = LOCKOUT_DURATION
            elif bars_held >= MAX_HOLD_BARS:
                net_pnl = ((current_close - entry_price) * lot_size * 100) - cost_per_trade
                balance += net_pnl; trade_log.append(net_pnl); trade_durations.append(bars_held); in_position = False; lockout_timer = LOCKOUT_DURATION
        else:
            if base_signals[i] == 1 and lockout_timer == 0:
                in_position = True
                entry_price = current_close
                if is_oos:
                    current_tp = entry_price + max(0.92, vol_scale * 1.8)
                    current_sl = entry_price - min(4.50, vol_scale * 2.2)
                else:
                    current_tp = entry_price + 0.92
                    current_sl = entry_price - 4.50
                bars_held = 0

        equity_curve.append(balance)

    return np.array(trade_log), np.array(equity_curve), np.array(trade_durations), timestamps