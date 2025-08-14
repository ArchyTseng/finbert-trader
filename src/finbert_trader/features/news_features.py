# news_features.py
# Module: NewsFeatureEngineer
# Purpose: Dedicated class for news sentiment feature engineering using FinBERT.
# Design: Single responsibility for news processing; handles chunks and text selection.
# Linkage: Called by FeatureEngineer to clean and compute sentiment.
# Robustness: Weighted score for nuanced sentiment; config-driven text_cols; increased noise std for low var; min_var threshold.
# Updates: Increased noise_std to 0.2; added min_variance=0.1, if below warn and amplify noise.
# Updates: Added compute_risk method using config.risk_prompt, similar to compute_sentiment but for risk (1-5 scale, weighted probs); reference from FinRL_DeepSeek (3: Risk Prompt, 4.3: risk_score 1-5 -> perturbation 0.9-1.1).

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import numpy as np
import logging
import re

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NewsFeatureEngineer:
    def __init__(self, config):
        """
        Introduction
        ------------
        Initialize the NewsFeatureEngineer with configuration parameters.
        Loads FinBERT tokenizer and model for sentiment/risk computation, sets batch size and text columns.
        Configures variance threshold and noise std for score adjustment.

        Parameters
        ----------
        config : object
            Configuration object containing batch_size, text_cols, etc.

        Notes
        -----
        - Preloads ProsusAI/finbert for financial text classification; sets to eval mode for inference.
        - min_variance (0.1) as threshold for low-variance detection in scores.
        - noise_std_base (0.3) for Gaussian noise injection in adjustments.
        - text_cols determines which news text fields to process (e.g., title, summary).
        """
        self.config = config  # Store config object for class-wide access
        self.batch_size = self.config.batch_size  # Batch size for efficient processing of news texts
        self.text_cols = self.config.text_cols  # List of text columns to use for sentiment/risk computation

        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")  # Load tokenizer for FinBERT model
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")  # Load pre-trained FinBERT model for sequence classification
        self.model.eval()  # Set model to evaluation mode to disable training behaviors like dropout

        self.min_variance = 0.1  # Threshold for acceptable variance in computed scores; below this triggers adjustment
        self.noise_std_base = 0.3  # Increased base std from 0.2 to 0.3 for stronger baseline noise in low-variance adjustments

    def clean_news_data(self, news_df):
        """
        Introduction
        ------------
        Clean the news DataFrame by dropping useless columns, renaming, normalizing dates, and purifying text columns.
        Handles FNSPID dataset specifics; returns empty DF on errors for pipeline resilience.

        Parameters
        ----------
        news_df : pd.DataFrame
            Input news DataFrame chunk to clean.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame, or empty if errors occur.

        Notes
        -----
        - Drops columns like 'Unnamed: 0', 'Url', etc.; renames 'Article' to 'Full_Text', 'Stock_symbol' to 'Symbol'.
        - Normalizes 'Date' to naive datetime (no timezone) and drops invalid dates.
        - Text cleaning: Removes HTML, normalizes whitespace, strips punctuation, lowers case; fills NaNs with ''.
        - Logs each step and errors for traceability.
        """
        try:
            drop_cols = ['Unnamed: 0', 'Url', 'Publisher', 'Author']    # Drop useless columns in FNSPID dataset
            for col in drop_cols:
                if col in news_df.columns:
                    news_df.drop(col, axis=1, inplace=True)  # Remove column if present to reduce data noise
                    logging.info(f"NF Module - Drop {col} column in news data")  # Log drop for auditing
            logging.info(f"NF Module - clean_news_data - Total columns of news_df : {news_df.shape[1]}")  # Log remaining columns after drops

            news_df.rename(columns={'Article': 'Full_Text', 'Stock_symbol': 'Symbol'}, inplace=True)  # Standardize column names for consistency
            news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce').dt.tz_localize(None).dt.normalize()  # Convert to datetime, remove timezone, normalize to date-only
            news_df.dropna(subset=['Date'], inplace=True)  # Drop rows with invalid dates to ensure temporal integrity

            def clean_text(text):
                if pd.isnull(text):
                    return ""  # Handle NaN early to avoid errors in re.sub
                text = re.sub(r'<[^>]+>', '', text) # Remove HTML tags
                text = re.sub(r'\s+', ' ', text)     # Normalize multiple whitespaces to single space
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation and special chars
                return text.strip().lower() # Trim whitespace and convert to lowercase for standardization

            text_cols = ['Full_Text', 'Article_title', 'Lsa_summary', 'Luhn_summary', 'Textrank_summary', 'Lexrank_summary']  # Text columns to clean (summaries and full text)
            for col in text_cols:
                if col in news_df.columns:
                    news_df[col] = news_df[col].apply(clean_text)  # Apply cleaning function to each text column
                    news_df[col].fillna('', inplace=True)  # Fill any remaining NaNs with empty string to prevent issues downstream
            
            logging.info(f"NF Module - clean_news_data - Cleaned news chunk: {len(news_df)} rows")  # Log final row count after cleaning
            return news_df  # Return the cleaned DataFrame
        except Exception as e:
            logging.error(f"NF Module - clean_news_data - Cleaning error in chunk: {e}")  # Log exception details for debugging
            return pd.DataFrame()  # Return empty DF on error to maintain pipeline flow without crash

    def filter_random_news(self, news_df):
        """
        Introduction
        ------------
        Filter news DataFrame to randomly sample one news item per Date-Symbol group.
        Reduces redundancy by keeping a representative sample per day per symbol.

        Parameters
        ----------
        news_df : pd.DataFrame
            Input news DataFrame with 'Date' and 'Symbol' columns.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with one row per Date-Symbol, or original if empty.

        Notes
        -----
        - Uses groupby and sample(n=1) with random_state=42 for reproducibility.
        - Warns if input is empty but returns it unchanged.
        - Reset index for clean output without group levels.
        """
        if news_df.empty:
            # Early warning and return if DF is empty to avoid unnecessary grouping
            logging.warning(f"NF Module - filter_random_news - Empty news_df at filtering random news step ")
        return news_df.groupby(['Date', 'Symbol']).sample(n=1, random_state=42).reset_index(drop=True)  # Group by Date-Symbol, sample 1 per group reproducibly, reset index

    def sentiment_batch_scores(self, texts, senti_mode='sentiment'):
        """
        Introduction
        ------------
        Compute batch sentiment or risk scores for a list of texts using FinBERT model.
        Converts softmax probabilities to weighted continuous scores in [1, 5] range, with mode-specific weighting.
        Adds random perturbation for variability and clamps to bounds.

        Parameters
        ----------
        texts : list of str
            List of text strings to score (e.g., news titles or summaries).
        senti_mode : str, optional
            Mode for scoring: 'sentiment' (positive high) or 'risk' (negative high) (default: 'sentiment').

        Returns
        -------
        np.ndarray
            Array of scores (float) for each text, in [1, 5].

        Notes
        -----
        - Uses FinBERT tokenizer with padding/truncation, max_length=512.
        - Weighted: For 'sentiment': pos*5 + neu*3 + neg*1; For 'risk': pos*1 + neu*3 + neg*5.
        - Perturbation multiplies by uniform [0.9, 1.1] for added variance.
        - Clamps to [1.0, 5.0]; raises ValueError for invalid mode.
        - Inference with no_grad for efficiency; moves to CPU numpy for return.
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)  # Tokenize batch texts with padding and truncation for model input
        # Compute without gradient
        with torch.no_grad():
            outputs = self.model(**inputs)  # Forward pass through FinBERT model for logits

        # Map the correct index for positive, negative, neutral
        id2label = self.model.config.id2label  # Get id-to-label mapping from model config
        label2id = {v: k for k, v in id2label.items()}  # Invert to label-to-id for indexing
        pos_idx = label2id['positive']  # Index for positive class
        neg_idx = label2id['negative']  # Index for negative class
        neu_idx = label2id['neutral']  # Index for neutral class
        # logging.info(f"NF Module - sentiment_batch_scores - pos_idx: {pos_idx}, neg_idx: {neg_idx}, neu_idx: {neu_idx}")  # Log indices for auditing
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)  # Apply softmax to logits for probabilities
        if senti_mode and senti_mode == 'sentiment':
            # Weighted score: pos*5 + neu*3 + neg*1 (continuous in 1-5) for sentiment mode (positive favored)
            scores = probs[:, pos_idx] * 5 + probs[:, neu_idx] * 3 + probs[:, neg_idx] * 1
        elif senti_mode and senti_mode == 'risk':
            # Weighted score: pos*1 + neu*3 + neg*5 (continuous in 1-5) for risk mode (negative emphasized)
            scores = probs[:, pos_idx] * 1 + probs[:, neu_idx] * 3 + probs[:, neg_idx] * 5
        else:
            logging.error(f"NF Module - sentiment_batch_scores - Unknown sentiment_mode: {senti_mode}")
            raise ValueError(f"NF Module - Unknown sentiment_mode: {senti_mode}")  # Error for unsupported mode
        
        # Perturbation: 0.9-1.1 to add slight randomness and increase score variability
        perturbation = torch.FloatTensor(np.random.uniform(0.9, 1.1, scores.shape)).to(scores.device)  # Generate uniform multiplier tensor on same device
        scores = scores * perturbation  # Apply perturbation to scores

        # Clamp to [1, 5] range to ensure bounds
        scores = torch.clamp(scores, min=1.0, max=5.0)  # Limit scores within valid range

        return scores.cpu().numpy()  # Move to CPU and convert to numpy array for return
    

    def compute_sentiment_risk_score(self, news_df, senti_mode='sentiment'):
        """
        Introduction
        ------------
        Compute sentiment or risk scores for news DataFrame using FinBERT on combined text columns.
        Handles batch processing, aggregation per Date/Symbol, and noise addition if variance is low.
        Supports fallback to neutral 3.0 for missing texts; clips scores to [1.0, 5.0].

        Parameters
        ----------
        news_df : pd.DataFrame
            News DataFrame with 'Date', 'Symbol', and text columns.
        senti_mode : str, optional
            Mode: 'sentiment' (positive high) or 'risk' (negative high) (default: 'sentiment').

        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame with 'Date', 'Symbol', and '{senti_mode}_score' column.

        Notes
        -----
        - Combines available text_cols into 'combined_text'; filters empty.
        - Batch computes scores via sentiment_batch_scores.
        - Aggregates mean score per Date/Symbol.
        - If variance < min_variance (0.1), amplifies noise_std (base 0.3 * 2.0).
        - Logs variance, adjustments, and fallbacks for monitoring.
        - Raises ValueError for invalid senti_mode.
        """
        senti_col = f"{senti_mode}_score"  # Dynamic column name based on mode
        logging.info(f"NF Module - compute_sentiment_risk_score - Computing {senti_col} for mode {senti_mode}")  # Log start of computation
        if news_df.empty:
            # Early return for empty DF to avoid processing
            logging.info(f"NF Module - compute_sentiment_risk_score - Empty news_df, returning DataFrame with 'Date', 'Symbol', '{senti_col}'")
            return pd.DataFrame(columns=['Date', 'Symbol', senti_col])
        
        missing_cols = [col for col in self.text_cols if col not in news_df.columns]  # Check for missing specified text columns
        if missing_cols:
            # Handle missing columns: Use available or fallback to neutral
            logging.warning(f"NF Module - compute_sentiment_risk_score - Missing text cols: {missing_cols}, using available")
            avail_cols = [col for col in self.text_cols if col in news_df.columns]  # Filter to existing columns
            if not avail_cols:
                if senti_mode == 'sentiment':
                    # Fallback: Assign neutral 3.0 for sentiment if no text available
                    logging.info("NF Module - compute_sentiment_risk_score - No valid text columns; returning neutral 3.0 sentiment scores")
                    return news_df.assign(sentiment_score=3.0)
                elif senti_mode == 'risk':
                    # Fallback: Assign neutral 3.0 for risk if no text available
                    logging.info("NF Module - compute_sentiment_risk_score - No valid text columns; returning neutral 3.0 risk scores")
                    return news_df.assign(risk_score=3.0)
                else:
                    raise ValueError("No valid sentiment mode. Expected senti_mode :  'sentiment' or 'risk'")  # Error for invalid mode
        else:
            avail_cols = self.text_cols  # Use all specified if present
        
        # Combine content in self.text_cols , default : Article_Title and Textrank_summary
        news_df[avail_cols] = news_df[avail_cols].fillna('')
        news_df['combined_text'] = news_df[avail_cols].apply(lambda row: ' '.join(str(x).strip() for x in row).strip(), axis=1)  # Concat text columns into single string per row
        news_df = news_df[news_df['combined_text'] != '']  # Filter out rows with empty combined text to avoid invalid inputs

        if news_df.empty:
            # Post-filter check: Return empty schema if all filtered out
            logging.info(f"NF Module - compute_sentiment_risk_score - Empty news_df, returning DataFrame with 'Date', 'Symbol', '{senti_col}'")
            return pd.DataFrame(columns=['Date', 'Symbol', senti_col])
        
        # Set local counter
        total_rows = len(news_df)
        logging.info(f"NF Module - compute_sentiment_risk_score - Starting {senti_mode} sentiment analysis for {total_rows} rows")
        # Compute total batch size for processing
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size
        
        sentiment_scores = []
        for batch_idx, i in enumerate(range(0, total_rows, self.batch_size)):
            # Compute current batch index
            batch_end = min(i + self.batch_size, total_rows)
            current_batch_size = batch_end - i
            
            # Compute senti/risk score per batch
            batch_texts = news_df['combined_text'].iloc[i:batch_end].tolist()
            batch_scores = self.sentiment_batch_scores(batch_texts, senti_mode)
            sentiment_scores.extend(batch_scores)
            
            # Compute progress and log
            processed_rows = batch_end
            remaining_rows = total_rows - processed_rows
            progress_percentage = processed_rows / total_rows
            
            # Log details
            logging.info(f"NF Module - {senti_mode} processing: Batch {batch_idx + 1}/{num_batches} (size: {current_batch_size}) ({progress_percentage:.1%}) - {processed_rows}/{total_rows} rows, {remaining_rows} remaining")
        
        news_df[senti_col] = sentiment_scores  # Assign computed scores to DF

        # Drop 'combined_text' before groupby
        if 'combined_text' in news_df.columns:
            news_df.drop('combined_text', axis=1, inplace=True)  # Remove temporary column to clean DF
            logging.info("NF Module - Dropped temporary 'combined_text' column")  # Log cleanup

        # Aggregate mean sentiment/risk per Date/Symbol
        news_df = news_df.groupby(['Date', 'Symbol'])[senti_col].mean().reset_index()  # Compute average score per group
        logging.info(f"NF Module - compute_sentiment_risk_score - Computed weighted sentiment for chunk using cols: {avail_cols}")  # Log used columns

        if not news_df.empty:
            score_var = news_df[senti_col].var()  # Calculate variance of scores
            noise_std = self.noise_std_base  # Base noise std from class init
            if score_var < self.min_variance:
                # Low variance: Amplify noise for better data diversity
                logging.warning(f"NF Module - compute_sentiment_risk_score - Low {senti_mode} variance ({score_var:.4f}) < {self.min_variance}; amplifying noise")
                noise_std *= 2.0  # Stronger amplify from 1.5 to 2.0 for better diversity
            # Keep the score range [1.0 , 5.0]
            news_df[senti_col] = np.clip(news_df[senti_col] + np.random.normal(0, noise_std, len(news_df)),
                                                 1.0, 5.0)  # Add Gaussian noise and clip to range
            final_var = news_df[senti_col].var()  # Recalculate final variance
            logging.info(f"NF Module - compute_sentiment_risk_score - Sentiment variance after adjustment: {final_var:.4f}")  # Log adjusted variance
        
        return news_df  # Return aggregated and adjusted DF
