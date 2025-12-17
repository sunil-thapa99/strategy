import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD
import datetime as dt
import pandas as pd
import os

# --- Configuration for MTF Testing ---
# We MUST download the lowest required timeframe: 1 minute.
instrument = INSTRUMENT_FX_METALS_XAU_USD
interval = dukascopy_python.INTERVAL_MIN_1  # Changed to the highest resolution (1-minute)
offer_side = dukascopy_python.OFFER_SIDE_BID
start_date = dt.datetime(2023, 1, 1) 
end_date = dt.datetime(2025, 12, 15) 
file_name = 'xauusd_1m_historical_data.csv'

def fetch_xauusd_data():
    """
    Downloads historical OHLVC data for the specified ticker.
    """
    print(f"-> Fetching data for {instrument} from {start_date.date()} to {end_date.date()} at {interval} interval...")
    try:
        # Note on yfinance: 1-minute data is often limited to the last 7 to 60 days, 
        # depending on the security and the server load.
        df = dukascopy_python.fetch(
            instrument=instrument, interval=interval, start=start_date, end=end_date, offer_side=offer_side
        )
        return df
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()

def main():
    # 1. Fetch the 1-minute data
    df = fetch_xauusd_data()

    if df.empty:
        print("Data extraction failed or returned an empty dataset. Exiting.")
        return

    # 2. Clean and prepare data
    print("\n--- Extracted 1-Minute Data Sample ---")
    print(df.head())
    print(f"\nTotal 1-minute bars extracted: {len(df)}")
    
    # 3. Save the data
    try:
        df.to_csv(file_name)
        print(f"\nSuccessfully saved 1-minute data to {os.path.abspath(file_name)}")
        
    except Exception as e:
        print(f"An error occurred while saving or resampling the file: {e}")

if __name__ == "__main__":
    main()