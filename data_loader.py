# data_loader.py
import pandas as pd
from config import DATA_URL

def load_historical_data():
    print("Fetching 5-year historical M15 XAUUSD data...")

    try:
        df = pd.read_csv(
            DATA_URL,
            parse_dates=['Time'],
            index_col='Time'
        )
    except Exception as e:
        print(f"Failed to load historical data: {e}")
        raise

    print(f"Success! Loaded {len(df):,} rows.")
    return df