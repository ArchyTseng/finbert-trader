# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: dev
#     language: python
#     name: python3
# ---

# %%
"""
Nasdaq-100 Benchmark Test
Simple test to verify Nasdaq-100 benchmark data fetching and visualization
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate project root path
if '__file__' in globals():
    project_root = os.path.join(os.path.dirname(__file__), '..')
else:
    # For interactive environments, use current working directory
    project_root = os.path.join(os.getcwd(), '..')

sys.path.insert(0, project_root)

# Import necessary libraries
from src.finbert_trader.trading_backtest import TradingBacktest
from src.finbert_trader.config_setup import ConfigSetup
from src.finbert_trader.config_trading import ConfigTrading

print("Nasdaq-100 Benchmark Test Ready!")
print("=" * 50)

# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""
Step 1: Test Nasdaq-100 Data Fetching
Direct test of benchmark data retrieval
"""
setup_config = ConfigSetup()

# Test date range (recent 2 years)
test_config = {
    'start': '2022-01-01',
    'end': '2023-12-31',
}

# Create a simple ConfigTrading instance for testing
config_trading = ConfigTrading(custom_config=test_config, upstream_config=setup_config)

# Initialize TradingBacktest
backtester = TradingBacktest(config_trading)

print("Testing Nasdaq-100 benchmark data fetching...")

print(f"Fetching Nasdaq-100 data from {config_trading.start} to {config_trading.end}...")

# Fetch benchmark data
try:
    benchmark_data = backtester._get_nasdaq100_benchmark()
    
    if benchmark_data is not None and not benchmark_data.empty:
        print("Nasdaq-100 data fetched successfully!")
        print(f"Data shape: {benchmark_data.shape}")
        print(f"Date range: {benchmark_data.index[0]} to {benchmark_data.index[-1]}")
        print(f"Price range: ${benchmark_data.min():.2f} to ${benchmark_data.max():.2f}")
        print(f"Starting price: ${benchmark_data.iloc[0]:.2f}")
        print(f"Ending price: ${benchmark_data.iloc[-1]:.2f}")
        
        # Calculate benchmark returns
        benchmark_returns = backtester._calculate_benchmark_returns(benchmark_data)
        if len(benchmark_returns) > 0:
            total_return = (benchmark_data.iloc[-1] / benchmark_data.iloc[0] - 1) * 100
            annualized_return = ((1 + total_return/100) ** (252/len(benchmark_returns)) - 1) * 100
            volatility = np.std(benchmark_returns) * np.sqrt(252) * 100
            
            print(f"\nBenchmark Performance Metrics:")
            print(f"  Total Return: {total_return:.2f}%")
            print(f"  Annualized Return: {annualized_return:.2f}%")
            print(f"  Annualized Volatility: {volatility:.2f}%")
            print(f"  Sharpe Ratio (assuming 0% risk-free rate): {annualized_return/volatility:.4f}")
        else:
            print("Could not calculate benchmark returns")
    else:
        print("Failed to fetch Nasdaq-100 data")
        
except Exception as e:
    print(f"Error fetching Nasdaq-100  {e}")

# %%
"""
Step 2: Visualize Nasdaq-100 Benchmark
Create visualization of the benchmark data
"""

if 'benchmark_data' in locals() and benchmark_data is not None and not benchmark_data.empty:
    print("\n" + "="*50)
    print("Visualizing Nasdaq-100 Benchmark Data")
    
    # Create normalized benchmark curve (starting at 1.0)
    normalized_benchmark = benchmark_data / benchmark_data.iloc[0]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(25, 12))
    fig.suptitle('Nasdaq-100 Benchmark Analysis', fontsize=16, fontweight='bold')
    
    # 1. Price Chart
    axes[0, 0].plot(benchmark_data.index, benchmark_data.values, linewidth=2, color='blue')
    axes[0, 0].set_title('Nasdaq-100 Price Chart (QQQ ETF)')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Normalized Performance (starting at 1.0)
    axes[0, 1].plot(normalized_benchmark.index, normalized_benchmark.values, linewidth=2, color='green')
    axes[0, 1].set_title('Normalized Performance (Starting Value = 1.0)')
    axes[0, 1].set_ylabel('Normalized Value')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    # 3. Daily Returns Distribution
    if len(benchmark_returns) > 0:
        axes[1, 0].hist(benchmark_returns * 100, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(np.mean(benchmark_returns) * 100, color='red', linestyle='--', 
                          label=f'Mean: {np.mean(benchmark_returns)*100:.3f}%')
        axes[1, 0].legend()
    
    # 4. Drawdown Analysis
    if len(benchmark_data) > 1:
        rolling_max = np.maximum.accumulate(benchmark_data.values)
        drawdown = (benchmark_data.values - rolling_max) / (rolling_max + 1e-8) * 100
        axes[1, 1].plot(benchmark_data.index, drawdown, linewidth=2, color='red')
        axes[1, 1].fill_between(benchmark_data.index, drawdown, 0, alpha=0.3, color='red')
        axes[1, 1].set_title('Drawdown Analysis')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print("Benchmark visualization completed!")
    
    # Save the plot
    plot_dir = 'plot_cache'
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'nasdaq100_benchmark_analysis.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

# %%
"""
Step 3: Test Benchmark Returns Calculation
Verify the benchmark returns calculation functionality
"""

print("\n" + "="*50)
print("Testing Benchmark Returns Calculation")

if 'benchmark_data' in locals() and benchmark_data is not None and not benchmark_data.empty:
    # Test returns calculation
    returns = backtester._calculate_benchmark_returns(benchmark_data)
    
    if len(returns) > 0:
        print("Benchmark returns calculation successful!")
        print(f"Number of returns: {len(returns)}")
        print(f"Average daily return: {np.mean(returns)*100:.4f}%")
        print(f"Return volatility: {np.std(returns)*100:.4f}%")
        print(f"Min daily return: {np.min(returns)*100:.4f}%")
        print(f"Max daily return: {np.max(returns)*100:.4f}%")
        
        # Verify calculation manually
        manual_returns = benchmark_data.pct_change().dropna().values
        if np.allclose(returns, manual_returns, rtol=1e-10):
            print("Returns calculation verified against manual calculation!")
        else:
            print("Returns calculation discrepancy detected")
    else:
        print("No returns calculated")

# %%
"""
Step 4: Simple Strategy vs Benchmark Comparison
Create a simple comparison with a mock strategy
"""

print("\n" + "="*50)
print("Simple Strategy vs Benchmark Comparison")

if ('benchmark_data' in locals() and benchmark_data is not None and not benchmark_data.empty and 
    'benchmark_returns' in locals() and len(benchmark_returns) > 0):
    
    # Create a simple mock strategy (20% more volatile than benchmark with same trend)
    np.random.seed(42)  # For reproducibility
    strategy_volatility_multiplier = 1.2
    strategy_returns = benchmark_returns + np.random.normal(0, np.std(benchmark_returns) * 0.1, len(benchmark_returns))
    strategy_returns = strategy_returns * strategy_volatility_multiplier
    
    # Calculate cumulative returns for both
    strategy_cumulative = np.cumprod(1 + strategy_returns)
    benchmark_cumulative = np.cumprod(1 + benchmark_returns)
    
    # Normalize to start at 1.0
    strategy_cumulative_normalized = strategy_cumulative / strategy_cumulative[0]
    benchmark_cumulative_normalized = benchmark_cumulative / benchmark_cumulative[0]
    
    # Create comparison plot
    plt.figure(figsize=(25, 12))
    plt.plot(range(len(strategy_cumulative_normalized)), strategy_cumulative_normalized, 
             label='Mock Strategy', linewidth=2, color='blue')
    plt.plot(range(len(benchmark_cumulative_normalized)), benchmark_cumulative_normalized, 
             label='Nasdaq-100 Benchmark', linewidth=2, color='orange')
    
    plt.title('Strategy vs Benchmark Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Normalized Cumulative Return (Starting Value = 1.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    # Add performance metrics
    strategy_total_return = (strategy_cumulative_normalized[-1] - 1) * 100
    benchmark_total_return = (benchmark_cumulative_normalized[-1] - 1) * 100
    strategy_sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
    benchmark_sharpe = (np.mean(benchmark_returns) / np.std(benchmark_returns)) * np.sqrt(252)
    
    plt.text(0.02, 0.98, f'Strategy Total Return: {strategy_total_return:.2f}%\n'
                        f'Benchmark Total Return: {benchmark_total_return:.2f}%\n'
                        f'Strategy Sharpe: {strategy_sharpe:.4f}\n'
                        f'Benchmark Sharpe: {benchmark_sharpe:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("Strategy vs Benchmark comparison completed!")
    print(f"Strategy Total Return: {strategy_total_return:.2f}%")
    print(f"Benchmark Total Return: {benchmark_total_return:.2f}%")
    print(f"Strategy Sharpe Ratio: {strategy_sharpe:.4f}")
    print(f"Benchmark Sharpe Ratio: {benchmark_sharpe:.4f}")
    
    # Save comparison plot
    plot_path = os.path.join('plot_cache', 'strategy_vs_benchmark_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")

# %%
"""
Step 5: Summary and Test Results
"""

print("\n" + "="*60)
print("NASDAQ-100 BENCHMARK TEST SUMMARY")
print("="*60)

success_count = 0
total_tests = 4

# Test 1: Data fetching
if 'benchmark_data' in locals() and benchmark_data is not None and not benchmark_data.empty:
    print("Test 1: Nasdaq-100 Data Fetching - PASSED")
    success_count += 1
else:
    print("Test 1: Nasdaq-100 Data Fetching - FAILED")

# Test 2: Returns calculation
if 'benchmark_returns' in locals() and len(benchmark_returns) > 0:
    print("Test 2: Returns Calculation - PASSED")
    success_count += 1
else:
    print("Test 2: Returns Calculation - FAILED")

# Test 3: Visualization
if 'normalized_benchmark' in locals():
    print("Test 3: Visualization - PASSED")
    success_count += 1
else:
    print("Test 3: Visualization - FAILED")

# Test 4: Strategy comparison
if 'strategy_cumulative_normalized' in locals():
    print("Test 4: Strategy Comparison - PASSED")
    success_count += 1
else:
    print("Test 4: Strategy Comparison - FAILED")

print(f"\nTest Results: {success_count}/{total_tests} tests passed")

if success_count == total_tests:
    print("\n ALL TESTS PASSED!")
    print("Nasdaq-100 benchmark functionality is working correctly!")
    print("Ready to integrate with full backtesting pipeline!")
else:
    print(f"\n{total_tests - success_count} tests failed!")
    print("Please check the error messages above and verify your internet connection.")
    print("Ensure yfinance package is properly installed: pip install yfinance")

print("\nOutput Files Generated:")
print("- plot_cache/nasdaq100_benchmark_analysis.png")
print("- plot_cache/strategy_vs_benchmark_comparison.png")

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)

# %%
