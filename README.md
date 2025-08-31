# FinBERT-Trader: A Localized Platform for Feature-Injected Financial Reinforcement Learning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#) [![Pixi](https://img.shields.io/badge/managed%20with-pixi-blue)](https://pixi.sh.io/)

## Project Overview

FinBERT-Trader is a lightweight, localized experimental platform designed for researching and testing the impact of feature injection, particularly financial news sentiment, in financial reinforcement learning (FRL). Built with a focus on **resource efficiency**, **reproducibility**, **modularity**, and **configurability**, it provides a practical foundation for FRL research on standard hardware without reliance on external, potentially unstable services.

This project was developed as part of a Master's program graduation requirement, aiming to create a substantial engineering project to facilitate a transition into the field of Computer Science and AI. The core motivation was to build a simplified, yet functional, pipeline to experimentally validate how injected features influence trading agent behavior and performance.

## Key Features & Innovations

*   **Dual-Channel Sentiment Feature Injection**: Transforms raw FinBERT outputs (positive, negative, neutral sentiment probabilities) into two distinct factors:
    *   `sentiment_score`: A weighted score emphasizing positive sentiment.
    *   `risk_score`: A weighted score emphasizing negative sentiment, used as a market risk proxy distinct from portfolio-level risk metrics.
*   **Dynamic Technical Indicator Thresholds**: Implements a mechanism to calculate indicator thresholds (e.g., RSI oversold/bought levels) dynamically on a *per-stock, per-episode* basis, allowing technical signal analysis to adapt to varying market conditions and stock characteristics.
*   **Modular & Configurable Architecture**: The platform is structured into four clear layers (Configuration, Data & Feature Engineering, RL Trading Agent, Experiment & Visualization), facilitating easy modification and extension. Comprehensive configuration files (`ConfigSetup`, `ConfigTrading`) control experiments and environment parameters.
*   **Local & Resource-Efficient**: Operates entirely offline using freely available data sources (`yfinance` for market data, FNSPID for news) and standard machine learning libraries, making it accessible for individual researchers and students.
*   **Integrated Experimentation Pipeline**: Includes a structured `ExperimentScheme` for defining and running comparative experiments (e.g., ablation studies) and a `VisualizeBacktest` module for analyzing performance.

## Architecture

The platform follows a four-layer architecture:

1.  **Configuration Layer**: Centralized configuration management using `ConfigSetup` (global settings) and `ConfigTrading` (environment-specific settings, inheriting from `ConfigSetup`).
2.  **Data & Feature Engineering Layer**: Modules (`DataResource`, `FeatureEngineer`, `StockFeatureEngineer`, `NewsFeatureEngineer`) handle data fetching, technical indicator calculation (via `TA-Lib`), and FinBERT-based sentiment analysis.
3.  **RL Trading Agent Layer**: Core RL components including the custom Gymnasium environment (`StockTradingEnv`) and the agent manager (`TradingAgent`), built upon Stable-Baselines3 (PPO as default).
4.  **Experiment & Visualization Layer**: Orchestrates experiments (`ExperimentScheme`), performs backtesting (`TradingBacktest`), and provides tools for result analysis and visualization (`VisualizeBacktest`).

## Getting Started

This project uses [Pixi](https://pixi.sh) for environment management and dependency resolution, ensuring a consistent and reproducible setup.

### Prerequisites

*   [Pixi](https://pixi.sh/latest/#installation): A package management tool that simplifies setting up development environments.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd finbert-trader
    ```
2.  **Initialize the Pixi environment:**
    Pixi will automatically read the `pyproject.toml` file and set up the environment with the correct Python version (>=3.11) and all required dependencies.
    ```bash
    pixi install
    ```

### Running Experiments

1.  **Activate the Pixi environment:**
    ```bash
    pixi shell
    ```
2.  **Configure your experiment:**
    Modify the `ConfigSetup` class or create configuration dictionaries to define your symbols, date ranges, indicators, and experiment parameters. Experiments can be defined within the `ExperimentScheme`.
3.  **Execute an experiment:**
    Run the main script or the specific experiment function you have defined. The exact command depends on your project's entry point structure (e.g., `python experiment_main.py` or calling a method on `ExperimentScheme`).


## Core Modules (Brief)

*   `stock_trading_env.py`: Defines the `StockTradingEnv` Gymnasium environment, including state/action spaces, reward calculation (`_calculate_strategy_reward`), dynamic thresholding (`_calculate_dynamic_ind_threshold`), and action interpretation (`_interpret_actions_strategy`).
*   `feature_engineer.py`: Orchestrates the feature engineering pipeline, including merging features from stocks and news.
*   `news_features.py`: Handles FinBERT model loading and sentiment score computation.
*   `stock_features.py`: Calculates technical indicators using TA-Lib.
*   `exper_scheme.py`: Defines and executes comparative experiments.
*   `trading_agent.py`: Manages the RL agent (training, loading, saving) using Stable-Baselines3.
*   `trading_backtest.py`: Runs backtests for trained agents.
*   `visualize_backtest.py`: Provides visualization tools for backtest results.

## Results & Findings

Experiments demonstrated that both dynamic technical indicators and FinBERT-derived sentiment/risk factors can individually improve trading performance metrics (e.g., Sharpe Ratio) compared to a baseline PPO agent. However, combining these factors directly sometimes led to underperformance, potentially due to conflicting signals and complexity in the reward function. This highlights the nuanced nature of multi-factor integration.

## Limitations & Future Work

*   **High Maximum Drawdown**: Observed significant drawdowns, indicating a need for stronger risk management mechanisms.
*   **Factor Interaction Complexity**: Combining technical and sentiment/risk signals can lead to conflicting guidance for the agent.
*   **Future Enhancements**:
    *   Implement more direct risk controls (e.g., dynamic drawdown penalties, hard stop-loss).
    *   Explore adaptive fusion mechanisms for multi-factor signals.
    *   Integrate real-time data streams and online LLM APIs.
    *   Investigate risk-sensitive RL algorithms (e.g., CPPO).

## Related Work

This project draws inspiration from and aims to complement frameworks like **FinRL**. While FinRL offers a comprehensive and powerful suite for FRL, FinBERT-Trader focuses specifically on providing a streamlined, lightweight environment for feature injection experiments, emphasizing local execution and modular design.

## Acknowledgements

This project utilizes and builds upon several excellent open-source libraries and datasets, including FinRL, Stable-Baselines3, TA-Lib, yfinance, Hugging Face Transformers (for FinBERT), PyTorch, and the FNSPID dataset.

## Author

[Shicheng Zeng]

## License

This project is licensed under the MIT License - see the LICENSE file for details.