"""
fetch_dax_index_data_last_26_days.py

Fetch last 26 business days of DAX (^GDAXI) index data
and save it to a CSV and Parquet file in data/dax/.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import BDay
import os

def fetch_dax_index_data(start_date: str, end_date: str, ticker: str = "^GDAXI") -> pd.DataFrame:
    print(f"Fetching data for ticker: {ticker} from {start_date} to {end_date}")
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=True
    )
    if not df.empty:
        df.index.name = "Date"
    return df

def main():
    dax_index_ticker = "^GDAXI"

    # 🚀 Dynamically get last 26 business days
    end_date = datetime.today().date()
    start_date = (end_date - BDay(26)).date()

    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()

    print(f"Attempting to fetch DAX index data for ticker {dax_index_ticker} from {start_date_str} to {end_date_str}.")

    dax_df = fetch_dax_index_data(start_date_str, end_date_str, ticker=dax_index_ticker)

    if dax_df.empty:
        print(f"No data returned for {dax_index_ticker} from {start_date_str} to {end_date_str}.")
        return

    print("\nFirst 3 rows of fetched data:")
    print(dax_df.head(3))
    print("\nLast 3 rows of fetched data:")
    print(dax_df.tail(3))
    print(f"\nTotal rows fetched: {len(dax_df)}")

    # ✅ Ensure output directory exists
    output_dir = "data/dax"
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Save files into data/dax/
    output_filename_base = os.path.join(output_dir, "dax_index_data_last26days")
    output_csv_file = f"{output_filename_base}.csv"
    output_parquet_file = f"{output_filename_base}.parquet"

    dax_df.to_csv(output_csv_file)
    print(f"\nSaved DAX index data to {output_csv_file}")

    try:
        dax_df.to_parquet(output_parquet_file)
        print(f"Saved DAX index data to {output_parquet_file} (Parquet format)")
    except ImportError:
        print("Could not save to Parquet. To enable Parquet, install pyarrow: pip install pyarrow")
    except Exception as e:
        print(f"An error occurred while saving to Parquet: {e}")

if __name__ == "__main__":
    main()
