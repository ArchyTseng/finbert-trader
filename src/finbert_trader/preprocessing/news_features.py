# news_feature_engineer.py
# Module: NewsFeatureEngineer
# Purpose: Dedicated class for news sentiment feature engineering using FinBERT.
# Design: Single responsibility for news processing; handles chunks and text selection.
# Linkage: Called by Preprocessing.process_news_chunks to clean and compute sentiment.
# Robustness: Weighted score for nuanced sentiment; config-driven text_cols.

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
        Initialize with config (e.g., 'text_cols', 'batch_size').
        Loads FinBERT model.
        """
        self.config = config
        self.batch_size = self.config.batch_size
        self.text_cols = self.config.text_cols
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()

    def clean_news_data(self, news_df):
        """
        Clean news DataFrame chunk.
        Input: news_df (pd.DataFrame chunk)
        Output: Cleaned news_df
        Logic: Drop irrelevant; rename; clean text; handle NaN.
        """
        try:
            if 'Unnamed: 0' in news_df.columns:
                news_df.drop('Unnamed: 0', axis=1, inplace=True)
            news_df.rename(columns={'Article': 'News_Text', 'Stock_symbol': 'Symbol'}, inplace=True)
            news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce')
            news_df.dropna(subset=['Date'], inplace=True)

            def clean_text(text):
                if pd.isnull(text):
                    return ""
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
                return text.lower()

            text_cols = ['News_Text', 'Article_title', 'Lsa_summary', 'Luhn_summary', 'Textrank_summary', 'Lexrank_summary']
            for col in text_cols:
                if col in news_df.columns:
                    news_df[col] = news_df[col].apply(clean_text)
                    news_df[col].fillna('', inplace=True)
            
            logging.info(f"Cleaned news chunk: {len(news_df)} rows")
            return news_df
        except Exception as e:
            logging.error(f"Cleaning error in chunk: {e}")
            return pd.DataFrame()

    def compute_sentiment(self, news_df):
        """
        Compute weighted sentiment scores per chunk.
        Input: cleaned news_df
        Output: news_df with 'sentiment_score' (float 1-5 range)
        Logic: Concat text_cols; batch infer probs; weighted_score = pos*5 + neu*3 + neg*1; aggregate mean.
        Robustness: Handle missing cols; drop empty text.
        """
        if news_df.empty:
            return pd.DataFrame()
        
        missing_cols = [col for col in self.text_cols if col not in news_df.columns]
        if missing_cols:
            logging.warning(f"Missing text cols: {missing_cols}, using available")
            avail_cols = [col for col in self.text_cols if col in news_df.columns]
            if not avail_cols:
                raise ValueError("No valid text columns for sentiment")
        else:
            avail_cols = self.text_cols
        
        news_df['combined_text'] = news_df[avail_cols].apply(lambda row: ' '.join(row).strip(), axis=1)
        news_df = news_df[news_df['combined_text'] != '']
        
        def sentiment_batch(texts):
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)  # [pos, neg, neutral]
            # Weighted score: pos*5 + neu*3 + neg*1 (continuous in 1-5)
            scores = probs[:, 0] * 5 + probs[:, 2] * 3 + probs[:, 1] * 1
            return scores.numpy()
        
        sentiment_scores = []
        for i in range(0, len(news_df), self.batch_size):
            batch_texts = news_df['combined_text'].iloc[i:i+self.batch_size].tolist()
            batch_scores = sentiment_batch(batch_texts)
            sentiment_scores.extend(batch_scores)
        
        news_df['sentiment_score'] = sentiment_scores

        # Drop 'combined_text' before groupby to clean up temp column and avoid KeyError later
        if 'combined_text' in news_df.columns:
            news_df.drop('combined_text', axis=1, inplace=True)
            logging.info("Dropped temporary 'combined_text' column")

        # Aggregate mean sentiment per Date/Symbol
        news_df = news_df.groupby(['Date', 'Symbol'])['sentiment_score'].mean().reset_index()
        logging.info(f"Computed weighted sentiment for chunk using cols: {avail_cols}")
        return news_df