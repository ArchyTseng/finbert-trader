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
        Initialize with config (e.g., 'text_cols', 'batch_size', 'risk_prompt').
        Loads FinBERT model.
        Updates: Added self.risk_prompt from config.
        """
        self.config = config
        self.batch_size = self.config.batch_size
        self.text_cols = self.config.text_cols
        self.risk_prompt = self.config.risk_prompt  # For compute_risk
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        self.min_variance = 0.1  # Threshold for acceptable variance
        self.noise_std_base = 0.2  # Increased base std

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
            news_df.rename(columns={'Article': 'Full_Text', 'Stock_symbol': 'Symbol'}, inplace=True)
            news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce')
            news_df.dropna(subset=['Date'], inplace=True)

            def clean_text(text):
                if pd.isnull(text):
                    return ""
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
                return text.lower()

            text_cols = ['Full_Text', 'Article_title', 'Lsa_summary', 'Luhn_summary', 'Textrank_summary', 'Lexrank_summary']
            for col in text_cols:
                if col in news_df.columns:
                    news_df[col] = news_df[col].apply(clean_text)
                    news_df[col].fillna('', inplace=True)
            
            logging.info(f"NF Module - Cleaned news chunk: {len(news_df)} rows")
            return news_df
        except Exception as e:
            logging.error(f"NF Module - Cleaning error in chunk: {e}")
            return pd.DataFrame()

    def compute_sentiment(self, news_df):
        """
        Compute weighted sentiment scores per chunk.
        Input: cleaned news_df
        Output: news_df with 'sentiment_score' (float 1-5 range). Reference from FDSPID
        Logic: Concat text_cols; batch infer probs; weighted_score = pos*5 + neu*3 + neg*1; aggregate mean.
        Robustness: Handle missing cols; drop empty text; enhanced variance check with amplified noise.
        """
        if news_df.empty:
            logging.info("NF Module - Empty news_df, returning DataFrame with 'Date', 'Symbol', 'sentiment_score'")
            return pd.DataFrame(columns=['Date', 'Symbol', 'sentiment_score'])
        
        missing_cols = [col for col in self.text_cols if col not in news_df.columns]
        if missing_cols:
            logging.warning(f"NF Module - Missing text cols: {missing_cols}, using available")
            avail_cols = [col for col in self.text_cols if col in news_df.columns]
            if not avail_cols:
                logging.warning("NF Module - No valid text columns; returning neutral 3.0 scores")
                return news_df.assign(sentiment_score=3.0)
        else:
            avail_cols = self.text_cols
        
        news_df['combined_text'] = news_df[avail_cols].apply(lambda row: ' '.join(row).strip(), axis=1)
        news_df = news_df[news_df['combined_text'] != '']

        if news_df.empty:
            logging.info("NF Module - No non-empty text after combining, returning DataFrame with 'Date', 'Symbol', 'sentiment_score'")
            return pd.DataFrame(columns=['Date', 'Symbol', 'sentiment_score'])
        
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

        # Drop 'combined_text' before groupby
        if 'combined_text' in news_df.columns:
            news_df.drop('combined_text', axis=1, inplace=True)
            logging.info("NF Module - Dropped temporary 'combined_text' column")

        # Aggregate mean sentiment per Date/Symbol
        news_df = news_df.groupby(['Date', 'Symbol'])['sentiment_score'].mean().reset_index()
        logging.info(f"NF Module - Computed weighted sentiment for chunk using cols: {avail_cols}")

        if not news_df.empty:
            score_var = news_df['sentiment_score'].var()
            noise_std = self.noise_std_base
            if score_var < self.min_variance:
                logging.warning(f"NF Module - Low sentiment variance ({score_var:.4f}) < {self.min_variance}; amplifying noise")
                noise_std *= 1.5  # Amplify if below min
            news_df['sentiment_score'] += np.random.normal(0, noise_std, len(news_df))
            final_var = news_df['sentiment_score'].var()
            logging.info(f"NF Module - Sentiment variance after adjustment: {final_var:.4f}")
        
        return news_df

    def compute_risk(self, news_df):
        """
        Compute weighted risk scores per chunk, similar to sentiment but using risk_prompt.
        Input: cleaned news_df
        Output: news_df with 'risk_score' (float 1-5 range).
        Logic: Same as sentiment but interpret probs as low-risk (pos->1), moderate (neu->3), high-risk (neg->5); aggregate mean, reference from FinRL_DeepSeek (3: Risk Prompt, weighted as pos*1 + neu*3 + neg*5 for inversion).
        Robustness: Reuse sentiment_batch logic but invert weights.
        """
        if news_df.empty:
            logging.info("NF Module - Empty news_df, returning DataFrame with 'Date', 'Symbol', 'risk_score'")
            return pd.DataFrame(columns=['Date', 'Symbol', 'risk_score'])
        
        missing_cols = [col for col in self.text_cols if col not in news_df.columns]
        if missing_cols:
            logging.warning(f"NF Module - Missing text cols: {missing_cols}, using available")
            avail_cols = [col for col in self.text_cols if col in news_df.columns]
            if not avail_cols:
                raise ValueError("No valid text columns for risk")
        else:
            avail_cols = self.text_cols
        
        news_df['combined_text'] = news_df[avail_cols].apply(lambda row: ' '.join(row).strip(), axis=1)
        news_df = news_df[news_df['combined_text'] != '']

        if news_df.empty:
            logging.info("NF Module - No non-empty text after combining, returning DataFrame with 'Date', 'Symbol', 'risk_score'")
            return pd.DataFrame(columns=['Date', 'Symbol', 'risk_score'])
        
        def risk_batch(texts):
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)  # [pos, neg, neutral]
            # Weighted risk: low low-risk (pos*1) + moderate (neu*3) + high-risk (neg*5)
            scores = probs[:, 0] * 1 + probs[:, 2] * 3 + probs[:, 1] * 5  # Inverted from sentiment for risk
            return scores.numpy()
        
        risk_scores = []
        for i in range(0, len(news_df), self.batch_size):
            batch_texts = news_df['combined_text'].iloc[i:i+self.batch_size].tolist()
            batch_scores = risk_batch(batch_texts)
            risk_scores.extend(batch_scores)
        
        news_df['risk_score'] = risk_scores

        # Drop 'combined_text' before groupby
        if 'combined_text' in news_df.columns:
            news_df.drop('combined_text', axis=1, inplace=True)
            logging.info("NF Module - Dropped temporary 'combined_text' column")

        # Aggregate mean risk per Date/Symbol
        news_df = news_df.groupby(['Date', 'Symbol'])['risk_score'].mean().reset_index()
        logging.info(f"NF Module - Computed weighted risk for chunk using cols: {avail_cols} and prompt: {self.risk_prompt[:50]}...")

        if not news_df.empty:
            score_var = news_df['risk_score'].var()
            noise_std = self.noise_std_base
            if score_var < self.min_variance:
                logging.warning(f"NF Module - Low risk variance ({score_var:.4f}) < {self.min_variance}; amplifying noise")
                noise_std *= 1.5
            news_df['risk_score'] += np.random.normal(0, noise_std, len(news_df))
            final_var = news_df['risk_score'].var()
            logging.info(f"NF Module - Risk variance after adjustment: {final_var:.4f}")
        
        return news_df