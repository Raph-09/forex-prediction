# main.py
import numpy as np
from data_loader import load_historical_data
from features import generate_features
from model import split_and_scale, train_and_calibrate
from backtester import run_backtest
import analytics

def main():
    # 1. Pipeline Ingestion & Feature Formulation
    raw_df = load_historical_data()
    features_df, X, y = generate_features(raw_df)
    
    # 2. Strict Boundary Isolation Scaling
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = split_and_scale(X, y)
    
    # 3. Model Engine Training & Threshold Calibration
    model, pct_cutoff_in_sample = train_and_calibrate(X_train_scaled, y_train)
    
    # 4. Out-of-Sample Validation Stream
    print("\nProcessing Advanced Production Out-of-Sample Matrices...")
    raw_probs_out_sample = model.predict_proba(X_test_scaled)[:, 1]
    oos_signals = np.where(raw_probs_out_sample >= pct_cutoff_in_sample, 1, 0)
    eval_prices_df = features_df.loc[X_test.index]
    
    # Execute Base Engine Backtest
    trade_log, equity_curve, durations, timestamps = run_backtest(
        eval_prices_df, oos_signals, is_oos=True
    )
    
    # =====================================================================
    # 5. CORE ENHANCED ANALYTICS INTERFACE PIPE
    # =====================================================================
    print("\n" + ("=" * 20) + " ADVANCED SYSTEM SUMMARY METRICS " + ("=" * 20))
    
    # Visual Matrix Verification Line
    print(f"\n--- [EQUITY CURVE ANALYSIS DIAGNOSTIC] ---")
    print(f"Starting OOS Capital Account State : ${equity_curve[0]:.2f}")
    print(f"Terminal OOS Capital Account State : ${equity_curve[-1]:.2f}")
    print(f"Total Net Point Yield Return Value : {((equity_curve[-1]-equity_curve[0])/equity_curve[0])*100:.2f}%")
    
    # Structural Analytical Passes
    analytics.run_trade_level_analysis(trade_log, durations)
    analytics.run_risk_and_robustness(trade_log, equity_curve)
    analytics.run_regime_performance(eval_prices_df, trade_log, timestamps)
    analytics.run_sensitivity_matrix(eval_prices_df, model, X_test_scaled)
    analytics.run_monte_carlo_simulation(trade_log)
    analytics.execute_walk_forward_validation()
    analytics.execute_paper_trading_link()
    print("\n" + "="*73)

if __name__ == "__main__":
    main()