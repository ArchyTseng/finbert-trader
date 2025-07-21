# data_resource.py
# Module: DataResource
# Purpose: Fetch and download raw data; optimized for large datasets.
# Updates: Now takes config instance in __init__; accesses params like config.symbols.
# Linkage: config passed from main; outputs to Preprocessing.

import yfinance as yf
import requests
import pandas as pd
import os
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataResource:
    def __init__(self, config):
        """
        Initialize with config dict containing symbols, start, end, chunksize (default 100000 for large files).
        Config example: {'symbols': ['AAPL'], 'start': '2010-01-01', 'end': '2023-12-31', 'chunksize': 100000}
        """
        self.config = config
        self.cache_dir = config.DATA_SAVE_DIR
        self.news_url = 'https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv'
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, symbol):
        return os.path.join(self.cache_dir, f"{symbol}_{self.config.start}_{self.config.end}.csv")

    def clean_yf_ohlcv(self, df):
        """
        Clean up the OHLCV data from yfinance.
        Input: DataFrame from yfinance
        Output: Cleaned DataFrame with columns ['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Volume']
        """
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = df.columns.str.strip().str.capitalize()    # Standadise columns format
        ordered_cols = ['Open', 'High', 'Low', 'Close', 'Volume']   # Set the target order
        df = df[ordered_cols]   # Reset the order of columns
        df.columns = [f'Adj_{col}' if col != 'Volume' else 'Volume' for col in df.columns]   # Reset the column name, e.g. "Adj_Open"
        return df

    def fetch_stock_data(self):
        """
        Fetch adjusted stock OHLCV data using yfinance.
        Input: None (uses self.config)
        Output: Dict of {symbol: DataFrame} with columns ['Date', 'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Volume', 'Symbol']
        Logic: Download, clean, standardize Date to timezone-naive; cache handling.
        Robustness: Use df['Date'].dt.tz is not None to safely check for timezone (works for both tz-aware and naive dtypes without AttributeError).
        """
        stock_data = {}
        for symbol in self.config.symbols:
            try:
                # Disign Cache Check
                cache_path = self.get_cache_path(symbol)

                if os.path.exists(cache_path):
                    logging.info(f"Loading cached data for {symbol}")
                    df = pd.read_csv(cache_path, parse_dates=["Date"])
                else:
                    logging.info(f"Downloading data for {symbol}")
                    df = yf.download(symbol,
                                    start=self.config.start,
                                    end=self.config.end,
                                    auto_adjust=True)
                    df = self.clean_yf_ohlcv(df)
                    df.reset_index(inplace=True)
                    # Standardize Date to timezone-naive (remove UTC if present)
                    if df['Date'].dt.tz is not None:
                        df['Date'] = df['Date'].dt.tz_convert(None)
                        logging.info(f"Removed timezone from Date column for {symbol}")
                    df.to_csv(cache_path, index=False)

                df['Symbol'] = symbol
                stock_data[symbol] = df
                logging.info(f"Downloaded stock data for {symbol}")
            except Exception as e:
                logging.error(f"Error downloading {symbol}: {e}")
        return stock_data

    def download_fnsqld_news_data(self, save_path='data_cache/nasdaq_exteral_data.csv'):
        """
        Download FNSPID news dataset using requests.
        Input: save_path (str, optional)
        Output: Path to saved CSV file
        Logic: GET request with stream for large files; check status.
        Robustness: Handle HTTP errors, ensure directory exists; log file size.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            response = requests.get(self.news_url, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
            logging.info(f"Downloaded news data to {save_path} (Size: {file_size:.2f} MB)")
            return save_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading news: {e}")
            return None

    def load_news_data(self, save_path='nasdaq_exteral_data.csv', chunksize=None):
        """
        Load large news CSV in chunks for efficiency, with cache check for filtered data.
        Input: save_path (str), chunksize (int, optional: from config or default 100000)
        Output: Generator yielding filtered pd.DataFrame chunks
        Logic: First check if filtered cache exists; if yes, load and yield as single chunk to skip heavy processing; else, fall back to original chunk loading, filtering, and saving.
        Robustness: Handle cache loading errors with fallback; ensure Date timezone-naive; log cache usage for traceability.
        Extensibility: Cache path based on symbols for multi-config support; can add config.use_cache flag in future.
        """
        # Config directory
        cache_path = os.path.join(self.cache_dir, os.path.basename(save_path))
        filtered_save_path = os.path.join(self.cache_dir, f"{'_'.join(self.config.symbols)}_filtered_news.csv")

        # Cache check for pre-filtered news to skip expensive operations
        if os.path.exists(filtered_save_path):
            logging.info(f"Loading from filtered cache: {filtered_save_path}")
            try:
                filtered_df = pd.read_csv(filtered_save_path, parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, utc=False))
                # Standardize Date timezone if needed
                if filtered_df['Date'].dt.tz is not None:   # Safe check: dt.tz is None for naive
                    filtered_df['Date'] = filtered_df['Date'].dt.tz_convert(None)
                    logging.info("Removed timezone from Date in filtered cache")
                # Yield the entire filtered_df as a single chunk (compatible with downstream chunk processing)
                yield filtered_df
                logging.info(f"Loaded {len(filtered_df)} rows from filtered cache")
                return  # Skip further processing
            except Exception as e:
                logging.error(f"Error loading filtered cache: {e}. Falling back to original processing.")
        
        # Original logic if no cache or loading failed
        if not os.path.exists(cache_path):
            logging.info("No cached news data found. Downloading...")
            downloaded_path = self.download_fnsqld_news_data(save_path=cache_path)
            if downloaded_path is None:
                logging.error("Download failed. Aborting load_news_data.")
                return

        # Init Chunksize and Chunk Reader
        chunksize = chunksize or self.config.chunksize
        try:
            reader = pd.read_csv(
                cache_path,
                chunksize=chunksize,
                low_memory=False,
                parse_dates=['Date'],
                dtype={'Stock_symbol': str},
                date_parser=lambda x: pd.to_datetime(x, utc=False)  # Explicitly parse without timezone
            )
        except Exception as e:
            logging.error(f"Error initializing CSV reader: {e}")
            yield pd.DataFrame()
            return

        # Load each chunk , filter and yield for saving and merging
        filtered_chunks = []

        try:
            for chunk in tqdm(reader, desc="Loading News Chunks"):
                if 'Stock_symbol' in chunk.columns:
                    chunk = chunk[chunk['Stock_symbol'].isin(self.config.symbols)]
                if not chunk.empty:
                    # Ensure Date is timezone-naive in chunks
                    if chunk['Date'].dt.tz is not None:
                        chunk['Date'] = chunk['Date'].dt.tz_convert(None)
                        logging.info("Removed timezone from Date in news chunk")
                    filtered_chunks.append(chunk.copy())
                    yield chunk
        except Exception as e:
            logging.error(f"Error during chunk processing: {e}")
            yield pd.DataFrame()

        # Save filtered news data
        if filtered_chunks:
            filtered_df = pd.concat(filtered_chunks)
            os.makedirs(os.path.dirname(filtered_save_path), exist_ok=True)
            filtered_df.to_csv(filtered_save_path, index=False)
            logging.info(f"Filtered news data saved to {filtered_save_path}")
