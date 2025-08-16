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
        Introduction
        ------------
        Initialize the DataResource class with configuration parameters.
        Sets up date ranges for training/testing, data cache directory, and FNSPID dataset URL.
        Ensures cache directory exists for data storage.

        Parameters
        ----------
        config : object
            Configuration object containing start/end dates, train_start_date, test_end_date, and RAW_DATA_DIR.

        Notes
        -----
        - Extracts date parameters from config for data fetching/splitting.
        - Creates raw_data_cache_dir if it doesn't exist using os.makedirs.
        - fnspid_url points to Hugging Face dataset for NASDAQ external news data.
        - Supports modular data management in RL pipelines.
        """
        self.config = config  # Store config object for class-wide access
        self.start = self.config.start  # Overall start date for data range
        self.end = self.config.end  # Overall end date for data range
        self.train_start_date = getattr(self.config, 'train_start_date', self.start)  # Start date specifically for training
        self.test_end_date = getattr(self.config, 'test_end_date', self.end)  # End date for testing (overwritten if config.end differs)
        self.raw_data_cache_dir = getattr(self.config, 'RAW_DATA_DIR', 'raw_data_cache')  # Directory for caching downloaded/processed data
        self.fnspid_url = 'https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv'  # URL for FNSPID NASDAQ news dataset on Hugging Face
        os.makedirs(self.raw_data_cache_dir, exist_ok=True)  # Create cache directory if it doesn't exist, with exist_ok to avoid errors

        self.use_symbol_name = getattr(self.config, 'use_symbol_name', True)

    def get_cache_path(self, symbol):
        """
        Introduction
        ------------
        Generate the cache file path for a stock symbol's data CSV.
        Uses raw_data_cache_dir, symbol, and date range (train_start_date or start, test_end_date).

        Parameters
        ----------
        symbol : str
            Stock symbol for the cache file name.

        Returns
        -------
        str
            Full path to the cache CSV file.

        Notes
        -----
        - Falls back to self.start if train_start_date is None; uses test_end_date for end.
        - Joins paths with os.path.join for cross-platform compatibility.
        - File name format: {symbol}_{start_date}_{end_date}.csv.
        """
        return os.path.join(self.raw_data_cache_dir, f"{symbol}_{self.train_start_date}_{self.test_end_date}.csv")  # Construct path with fallback dates for robustness; ensures unique file per symbol and range

    def clean_yf_ohlcv(self, df, symbol):
        """
        Introduction
        ------------
        Clean Yahoo Finance OHLCV DataFrame for a specific symbol.
        Standardizes column names, reorders columns, adds adjusted prefixes and symbol suffixes, and filters invalid rows.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with OHLCV columns, possibly multi-indexed.
        symbol : str
            Stock symbol for suffixing columns.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with standardized columns and filtered rows.

        Notes
        -----
        - Handles multi-index by flattening to level 0.
        - Columns standardized to capitalized, stripped; reordered to ['Open', 'High', 'Low', 'Close', 'Volume'].
        - Renames to 'Adj_{col}_{symbol}' (except Volume as 'Volume_{symbol}').
        - Filters rows where Adj_Close_{symbol} <= 0 to prevent invalid data in downstream processes.
        - Logs filtered row count for data quality monitoring.
        - Uses deep copy to preserve original DF.
        """
        df_original = df.copy(deep=True)  # Deep copy to avoid modifying the input DF
        df = df_original.copy()  # Working copy for cleaning operations
        try:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)  # Flatten multi-index columns to single level if present
            
            df.columns = df.columns.str.strip().str.capitalize()    # Standardize columns: strip whitespace and capitalize for consistency
            ordered_cols = ['Open', 'High', 'Low', 'Close', 'Volume']   # Set the target order for OHLCV columns
            df = df[ordered_cols]   # Reset the order of columns to ensure logical sequence
            df.columns = [f'Adj_{col}_{symbol}' if col != 'Volume' else f'Volume_{symbol}' for col in df.columns]   # Reset the column name, e.g. "Adj_Open_AAPL" for multi-stock compatibility

            # Filter invalid rows for robustness (prevent divide by zero in env)
            df = df[df[f"Adj_Close_{symbol}"] > 0]  # Remove rows with non-positive Adj_Close to avoid invalid calculations downstream
            if len(df) < len(df_original):
                logging.info(f"DR Module - clean_yf_ohlcv - Filtered out {len(df_original) - len(df)} rows with Adj_Close <= 0")  # Log filtered rows for data integrity tracking
            return df  # Return the cleaned and filtered DF
        except Exception as e:
            logging.warning(f"DR Module - clean_yf_ohlcv - Error cleaning OHLCV data for {symbol}: {e}")  # Log any errors during cleaning)

    def fetch_stock_data(self):
        """
        Introduction
        ------------
        Fetch or load cached stock OHLCV data for configured symbols from Yahoo Finance.
        Checks cache first; downloads with retry if missing, cleans data, and saves to cache.
        Handles date ranges, column standardization, and error logging per symbol.

        Parameters
        ----------
        None (uses self.config.symbols and date attributes).

        Returns
        -------
        dict
            Dictionary of pd.DataFrame keyed by symbol, each with cleaned stock data and 'Date' column.

        Notes
        -----
        - Prioritizes cache loading from raw_data_cache_dir; parses dates on load.
        - Downloads via yf.download with 3 retries (5s sleep) on failures.
        - Cleans via clean_yf_ohlcv; ensures 'Date' as datetime64[ns] without timezone.
        - Drops redundant 'Symbol' if present; resets index to include 'Date'.
        - Logs per-symbol progress, errors, and final dict type for traceability.
        """
        stock_data_dict = {}  # Initialize dict to store DataFrames per symbol
        for symbol in self.config.symbols:
            # Process each symbol in config
            try:
                # Design Cache Check
                cache_path = self.get_cache_path(symbol)  # Generate dynamic cache path for symbol

                if os.path.exists(cache_path):
                    # Cache hit: Load existing data to avoid redownload
                    logging.info("=========== Start to load stock data ===========")
                    logging.info(f"DR Module - fetch_stock_data - Loading cached data for {symbol}")
                    df = pd.read_csv(cache_path, parse_dates=["Date"])  # Load CSV with date parsing
                else:
                    # Cache miss: Proceed to download
                    logging.info("=========== Start to fetch stock data ===========")
                    logging.info(f"DR Module - fetch_stock_data - Downloading data for {symbol}")
                    start = self.train_start_date  # Fallback to overall start if train_start_date None
                    logging.info(f"DR Module - fetch_stock_data - {symbol} start date: {start}")
                    end = self.test_end_date  # Use test_end_date (note: potential config redundancy)
                    logging.info(f"DR Module - fetch_stock_data - {symbol} end date: {end}")
                    for attempt in range(3):  # Retry loop for network issues to enhance reliability
                        try:
                            df = yf.download(symbol,
                                             start=start,
                                             end=end,
                                             auto_adjust=True)  # Download adjusted OHLCV data
                            break  # Exit loop on success
                        except Exception as e:
                            logging.warning(f"DR Module - fetch_stock_data - Download attempt {attempt+1} failed for {symbol}: {e}; retrying in 5s")  # Log retry warning
                            time.sleep(5)  # Delay before next attempt
                    else:
                        raise RuntimeError(f"Failed to download {symbol} after 3 attempts")  # Raise after max retries
                    df = self.clean_yf_ohlcv(df, symbol)  # Clean downloaded data (standardize columns, filter invalid)
                    if 'Symbol' in df.columns:
                        df.drop('Symbol', axis=1, inplace=True)  # Drop redundant Symbol column if present
                        logging.info(f"DR Module - fetch_stock_data - Dropped 'Symbol' column for {symbol}")  # Log drop for auditing

                    df.reset_index(inplace=True)    # Set 'Date' as a column instead of index for consistency
                    # Ensure Date is datetime64[ns] without tz
                    if df['Date'].dtype != 'datetime64[ns]':
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None).dt.normalize()  # Normalize to naive date, coerce errors
                    logging.info(f"DR Module - fetch_stock_data - Stock {symbol} Date dtype: {df['Date'].dtype}")  # Log final dtype for verification
                    df.to_csv(cache_path, index=False)  # Save cleaned data to cache for future use

                stock_data_dict[symbol] = df  # Store cleaned DF in dict
                logging.info(f"DR Module - fetch_stock_data - Successfully download stock data for {symbol}")  # Log success per symbol
            except Exception as e:
                logging.error(f"DR Module - fetch_stock_data - Error downloading {symbol}: {e}")  # Log any unhandled errors
        logging.info(f"DR Module - fetch_stock_data - Return stock data for {self.config.symbols} as {type(stock_data_dict)}")  # Log final return type and symbols
        return stock_data_dict  # Return dict of all fetched/cleaned stock data

    def download_fnsqld_news_data(self, save_path='raw_data_cache/nasdaq_exteral_data.csv'):
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
            logging.info(f"DR Module - _fnsqld_news_data - Downloaded news data to {save_path} (Size: {file_size:.2f} MB)")
            return save_path
        except requests.exceptions.RequestException as e:
            logging.error(f"DR Module - _fnsqld_news_data - Error downloading news: {e}")
            return None

    def cache_path_config(self, base_path='nasdaq_exteral_data.csv'):
        """
        Introduction
        ------------
        Configure cache paths for news data: original and filtered by symbols and date range.
        Generates paths in raw_data_cache_dir; logs for traceability.

        Parameters
        ----------
        base_path : str, optional
            Base filename for original cache (default: 'nasdaq_exteral_data.csv').

        Returns
        -------
        tuple
            (original_cache_path: str, filtered_cache_path: str)

        Notes
        -----
        - Uses os.path.basename for original path to extract filename.
        - Filtered path: Joins symbols with '_', appends dates (fallback to start/end if train/test None).
        - Logs both paths for debugging cache operations.
        """
        cache_path = os.path.join(self.raw_data_cache_dir, os.path.basename(base_path))  # Construct original cache path using basename for filename isolation
        if self.use_symbol_name:
            symbols = self.config.symbols
            filtered_cache_path = os.path.join(self.raw_data_cache_dir, f"{'_'.join(symbols)}_{self.train_start_date}_{self.test_end_date}_news.csv")  # Build filtered path with symbols joined and date range (fallback for None dates)
        else:
            filtered_cache_path = os.path.join(self.raw_data_cache_dir, f"All_symbols_{self.train_start_date}_{self.test_end_date}_news.csv")  # Build filtered path with symbols joined and date range (fallback for None dates)
        logging.info(f"DR Module - cache_path_config - Cache path: {cache_path}, Filtered cache path: {filtered_cache_path}")  # Log paths for monitoring and debugging
        return cache_path, filtered_cache_path  # Return tuple of paths for use in fetching/saving

    def load_news_data(self, cache_path, filtered_cache_path, chunksize=None):
        """
        Introduction
        ------------
        Load news data from cache or download if missing, process in chunks, filter by date and symbols, and yield chunks.
        Prioritizes filtered cache for efficiency; falls back to original cache/download, cleans timezone, and saves filtered data.

        Parameters
        ----------
        cache_path : str
            Path to original news cache CSV.
        filtered_cache_path : str
            Path to filtered news cache CSV (by symbols and dates).
        chunksize : int, optional
            Chunk size for reading CSV; defaults to self.config.chunksize.

        Returns
        -------
        generator
            Yields pd.DataFrame chunks of filtered news data.

        Notes
        -----
        - If filtered cache exists, loads and yields as single chunk.
        - Downloads via download_fnsqld_news_data if original cache missing.
        - Cleans 'Date' to timezone-naive datetime64[ns].
        - Filters chunks to train_start_date to test_end_date and config.symbols.
        - Saves concatenated filtered chunks to filtered_cache_path.
        - Uses tqdm for progress; logs errors and steps for debugging.
        - Yields empty DF on errors to maintain flow.
        """
        if cache_path is None or filtered_cache_path is None:
            # Early exit if paths missing to prevent invalid operations
            logging.error("DR Module - load_news_data - Cache path is needed. Filtered cache path is needed.")
            return
        def clean_chunk_tz(chunk):
            """
            Clean up the Date column in a chunk to ensure it's timezone-naive.
            """
            if 'Date' in chunk.columns:
                chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce').dt.tz_localize(None) .dt.normalize()  # Convert to naive datetime, normalize to date-only, coerce errors
                logging.info("DR Module - load_news_data - Removed timezone from Date in chunk")  # Log timezone cleanup
            else:
                logging.info("DR Module - load_news_data - 'Date' column not found in chunk")  # Log if Date missing
            return chunk  # Return cleaned chunk

        # Cache check for pre-filtered news to skip expensive operations
        if os.path.exists(filtered_cache_path):
            # Filtered cache hit: Load directly for efficiency
            logging.info("=========== Start to load news data ===========")
            logging.info(f"DR Module - load_news_data - Loading from filtered cache: {filtered_cache_path}")
            try:
                filtered_df = pd.read_csv(filtered_cache_path, parse_dates=['Date'], dtype={'Stock_symbol': str})  # Load with date parsing and string dtype for symbol
                # Safe check: dtype is 'datetime64[ns]' without tz
                if filtered_df['Date'].dtype != 'datetime64[ns]':   
                    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='ISO8601').dt.tz_localize(None)  # Force ISO format and remove tz if needed
                    logging.info("DR Module - load_news_data - Removed timezone from Date in filtered cache")  # Log adjustment
                # Yield the entire filtered_df as a single chunk (compatible with downstream chunk processing)
                yield filtered_df  # Yield full DF as one chunk
                logging.info(f"DR Module - load_news_data - Loaded {len(filtered_df)} rows from filtered cache")  # Log loaded rows
                return  # Skip further processing after yielding
            except Exception as e:
                logging.error(f"DR Module - load_news_data - Error loading filtered cache: {e}. Falling back to original processing.")  # Log error and fallback
        
        # Original logic if no cache or loading failed
        if not os.path.exists(cache_path):
            # Original cache miss: Proceed to download
            logging.warning("DR Module - load_news_data - No cached news data found. Downloading...")
            logging.info("=========== Start to download FNSPID Raw Data ===========")
            downloaded_path = self.download_fnsqld_news_data(save_path=cache_path)  # Download and save raw data
            if downloaded_path is None:
                logging.error("DR Module - load_news_data - Download failed. Aborting load_news_data.")  # Log download failure
                return  # Exit if download fails

        # Initial Chunksize and Chunk Reader
        chunksize = chunksize or self.config.chunksize  # Use provided or config chunksize
        logging.info(f"DR Module - load_news_data - Chunksize: {chunksize}")  # Log chunksize for monitoring
        try:
            chunks = pd.read_csv(
                cache_path,
                chunksize=chunksize,
                low_memory=False,
                parse_dates=['Date'],
                dtype={'Stock_symbol': str},
            )  # Initialize chunk reader with low_memory off, date parsing, and str dtype for symbol
            logging.info(f"DR Module - load_news_data - Loaded chunks from {cache_path}")  # Log reader init
        except Exception as e:
            logging.error(f"DR Module - load_news_data - Error initializing CSV reader: {e}")  # Log reader error
            yield pd.DataFrame()  # Yield empty DF on error to prevent crash
            return  # Exit after error

        # Load each chunk , filter and yield for saving and merging
        filtered_chunks = []  # List to collect filtered chunks for final save
        logging.info(f"DR Module - load_news_data - Start to filter news data")  # Log start of filtering
        try:
            for chunk in tqdm(chunks, desc="Loading News Chunks"):
                # Process each chunk with progress bar
                chunk = clean_chunk_tz(chunk)  # Clean timezone in chunk
                chunk = chunk[(chunk['Date'] >= pd.to_datetime(self.train_start_date, errors='coerce')) & (chunk['Date'] <= pd.to_datetime(self.test_end_date, errors='coerce'))]  # Filter to global date range
                logging.info(f"DR Module - load_news_data - Chunk filtered to global date range: {self.train_start_date} to {self.test_end_date}, {len(chunk)} rows")  # Log post-date filter size
                if 'Stock_symbol' in chunk.columns:
                    chunk = chunk[chunk['Stock_symbol'].isin(self.config.symbols)]  # Filter to configured symbols
                if not chunk.empty:
                    filtered_chunks.append(chunk.copy())  # Collect non-empty filtered chunk
                    yield chunk  # Yield filtered chunk for downstream use
        except Exception as e:
            logging.error(f"DR Module - load_news_data - Error during chunk processing: {e}")  # Log processing error
            yield pd.DataFrame()  # Yield empty on error

        # Save filtered news data
        if filtered_chunks:
            # If chunks collected, concat and save
            filtered_df = pd.concat(filtered_chunks)  # Concatenate all filtered chunks
            os.makedirs(os.path.dirname(filtered_cache_path), exist_ok=True)  # Ensure directory for save
            filtered_df.to_csv(filtered_cache_path, index=False)  # Save without index
            logging.info(f"DR Module - load_news_data - Filtered news data saved to {filtered_cache_path}")  # Log save success