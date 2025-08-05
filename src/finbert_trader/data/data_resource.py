# data_resource.py
# Module: DataResource
# Purpose: Fetch and download raw data; optimized for large datasets.
# Updates: Now takes config instance in __init__; accesses params like config.symbols.
# Linkage: config passed from main; outputs to Preprocessing.
# Updates: Minor robustness enhancement: added retry for yfinance download (3 times) to handle network instability; no major changes as this module is stable.

import yfinance as yf
import requests
import pandas as pd
import os
from tqdm import tqdm
import logging
import time  # For retry delay

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataResource:
    def __init__(self, config):
        """
        Initialize with config dict containing symbols, start, end, chunksize (default 100000 for large files).
        Config example: {'symbols': ['AAPL'], 'start': '2010-01-01', 'end': '2023-12-31', 'chunksize': 100000}
        """
        self.config = config
        self.start = self.config.start
        self.test_end_date = self.config.end
        self.train_start_date = self.config.train_start_date
        self.test_end_date = self.config.test_end_date
        self.cache_dir = config.DATA_SAVE_DIR
        self.fnspid_url = 'https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv'
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, symbol):
        return os.path.join(self.cache_dir, f"{symbol}_{self.train_start_date if self.train_start_date else self.start}_{self.test_end_date if self.test_end_date else self.test_end_date}.csv")

    def clean_yf_ohlcv(self, df, symbol):
        """
        Clean up the OHLCV data from yfinance.
        Input: DataFrame from yfinance
        Output: Cleaned DataFrame with columns ['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Volume']
        Logic: Standardize columns; filter invalid prices (Adj_Close > 0) for trading stability.
        """
        df_original = df.copy(deep=True)
        df = df_original.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = df.columns.str.strip().str.capitalize()    # Standadise columns format
        ordered_cols = ['Open', 'High', 'Low', 'Close', 'Volume']   # Set the target order
        df = df[ordered_cols]   # Reset the order of columns
        df.columns = [f'Adj_{col}_{symbol}' if col != 'Volume' else f'Volume_{symbol}' for col in df.columns]   # Reset the column name, e.g. "Adj_Open_AAPL"

        # Filter invalid rows for robustness (prevent divide by zero in env)
        df = df[df[f"Adj_Close_{symbol}"] > 0]
        if len(df) < len(df_original):
            logging.info(f"DR Module - Filtered out {len(df_original) - len(df)} rows with Adj_Close <= 0")
        return df

    def fetch_stock_data(self):
        """
        Fetch adjusted stock OHLCV data using yfinance.
        Input: None (uses self.config)
        Output: Dict of {symbol: DataFrame} with columns ['Date', 'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Volume', 'Symbol']
        Logic: Download, clean, standardize Date to timezone-naive; cache handling.
        Robustness: Use df['Date'].dt.tz is not None to safely check for timezone (works for both tz-aware and naive dtypes without AttributeError); added retry (3 times) for download stability.
        """
        stock_data_dict = {}
        for symbol in self.config.symbols:
            try:
                # Design Cache Check
                cache_path = self.get_cache_path(symbol)

                if os.path.exists(cache_path):
                    logging.info("=========== Start to load stock data ===========")
                    logging.info(f"DR Module - Loading cached data for {symbol}")
                    df = pd.read_csv(cache_path, parse_dates=["Date"])
                else:
                    logging.info("=========== Start to fetch stock data ===========")
                    logging.info(f"DR Module - Downloading data for {symbol}")
                    start = self.train_start_date if self.train_start_date else self.start
                    logging.info(f"DR Module - {symbol} start date: {start}")
                    end = self.test_end_date if self.test_end_date else self.test_end_date
                    logging.info(f"DR Module - {symbol} end date: {end}")
                    for attempt in range(3):  # Retry loop for network issues
                        try:
                            df = yf.download(symbol,
                                             start=start,
                                             end=end,
                                             auto_adjust=True)
                            break
                        except Exception as e:
                            logging.warning(f"DR Module - Download attempt {attempt+1} failed for {symbol}: {e}; retrying in 5s")
                            time.sleep(5)
                    else:
                        raise RuntimeError(f"Failed to download {symbol} after 3 attempts")
                    df = self.clean_yf_ohlcv(df, symbol)
                    if 'Symbol' in df.columns:
                        df.drop('Symbol', axis=1, inplace=True)
                        logging.info(f"DR Module - Dropped 'Symbol' column for {symbol}")

                    df.reset_index(inplace=True)    # Set 'Date' as a column instead of index
                    # Ensure Date is datetime64[ns] without tz
                    if df['Date'].dtype != 'datetime64[ns]':
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None).dt.normalize()
                    logging.info(f"DR Module - Stock {symbol} Date dtype: {df['Date'].dtype}")
                    df.to_csv(cache_path, index=False)

                stock_data_dict[symbol] = df
                logging.info(f"DR Module - Successfully download stock data for {symbol}")
            except Exception as e:
                logging.error(f"DR Module - Error downloading {symbol}: {e}")
        logging.info(f"DR Module - Return stock data for {self.config.symbols} as {type(stock_data_dict)}")
        return stock_data_dict

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
            response = requests.get(self.fnspid_url, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
            logging.info(f"DR Module - Downloaded news data to {save_path} (Size: {file_size:.2f} MB)")
            return save_path
        except requests.exceptions.RequestException as e:
            logging.error(f"DR Module - Error downloading news: {e}")
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
        filtered_save_path = os.path.join(self.cache_dir, f"{'_'.join(self.config.symbols)}_{self.train_start_date if self.train_start_date else self.start}_{self.test_end_date if self.test_end_date else self.test_end_date}_news.csv")

        def clean_chunk_tz(chunk):
            """
            Clean up the Date column in a chunk to ensure it's timezone-naive.
            """
            if 'Date' in chunk.columns:
                chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce').dt.tz_localize(None) .dt.normalize() 
                logging.info("DR Module - Removed timezone from Date in chunk")
            else:
                logging.info("DR Module - 'Date' column not found in chunk")
            return chunk


        # Cache check for pre-filtered news to skip expensive operations
        if os.path.exists(filtered_save_path):
            logging.info("=========== Start to load news data ===========")
            logging.info(f"DR Module - Loading from filtered cache: {filtered_save_path}")
            try:
                filtered_df = pd.read_csv(filtered_save_path, parse_dates=['Date'], dtype={'Stock_symbol': str})
                # Safe check: dtype is 'datetime64[ns]' without tz
                if filtered_df['Date'].dtype != 'datetime64[ns]':   
                    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='ISO8601').dt.tz_localize(None)
                    logging.info("DR Module - Removed timezone from Date in filtered cache")
                # Yield the entire filtered_df as a single chunk (compatible with downstream chunk processing)
                yield filtered_df
                logging.info(f"DR Module - Loaded {len(filtered_df)} rows from filtered cache")
                return  # Skip further processing
            except Exception as e:
                logging.error(f"DR Module - Error loading filtered cache: {e}. Falling back to original processing.")
        
        # Original logic if no cache or loading failed
        if not os.path.exists(cache_path):
            logging.info("DR Module - No cached news data found. Downloading...")
            logging.info("=========== Start to download FNSPID Raw Data ===========")
            downloaded_path = self.download_fnsqld_news_data(save_path=cache_path)
            if downloaded_path is None:
                logging.error("DR Module - Download failed. Aborting load_news_data.")
                return

        # Initial Chunksize and Chunk Reader
        chunksize = chunksize or self.config.chunksize
        try:
            chunks = pd.read_csv(
                cache_path,
                chunksize=chunksize,
                low_memory=False,
                parse_dates=['Date'],
                dtype={'Stock_symbol': str},
            )
        except Exception as e:
            logging.error(f"DR Module - Error initializing CSV reader: {e}")
            yield pd.DataFrame()
            return

        # Load each chunk , filter and yield for saving and merging
        filtered_chunks = []
        try:
            for chunk in tqdm(chunks, desc="Loading News Chunks"):
                chunk = clean_chunk_tz(chunk)
                chunk = chunk[(chunk['Date'] >= pd.to_datetime(self.train_start_date, errors='coerce')) & (chunk['Date'] <= pd.to_datetime(self.test_end_date, errors='coerce'))]
                logging.info(f"DR Module - Chunk filtered to global date range: {self.train_start_date} to {self.test_end_date}, {len(chunk)} rows")
                if 'Stock_symbol' in chunk.columns:
                    chunk = chunk[chunk['Stock_symbol'].isin(self.config.symbols)]
                if not chunk.empty:
                    filtered_chunks.append(chunk.copy())
                    yield chunk
        except Exception as e:
            logging.error(f"DR Module - Error during chunk processing: {e}")
            yield pd.DataFrame()

        # Save filtered news data
        if filtered_chunks:
            filtered_df = pd.concat(filtered_chunks)
            os.makedirs(os.path.dirname(filtered_save_path), exist_ok=True)
            filtered_df.to_csv(filtered_save_path, index=False)
            logging.info(f"DR Module - Filtered news data saved to {filtered_save_path}")