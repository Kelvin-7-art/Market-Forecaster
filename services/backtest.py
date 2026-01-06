"""
Backtesting service for Market Forecaster.
Implements basic backtesting for trading signals.
"""

import pandas as pd
import numpy as np


def run_backtest(df: pd.DataFrame, initial_capital: float = 10000) -> dict:
    """
    Run a backtest on the signal-based strategy.
    
    Strategy:
    - Go long when signal is BUY
    - Exit when signal is SELL
    - Stay in position during HOLD if already in
    
    Args:
        df: DataFrame with signals and price data
        initial_capital: Starting capital
    
    Returns:
        Dictionary with backtest results
    """
    if df.empty or 'Signal' not in df.columns:
        return {
            'equity_curve': pd.DataFrame(),
            'trades': [],
            'stats': {}
        }
    
    df = df.copy().reset_index(drop=True)
    
    capital = initial_capital
    position = 0
    shares = 0
    entry_price = 0
    
    equity_curve = []
    trades = []
    
    for idx, row in df.iterrows():
        date = row['Date']
        price = row['Close']
        signal = row['Signal']
        
        if signal == 'BUY' and position == 0:
            shares = capital / price
            entry_price = price
            position = 1
            capital = 0
            trades.append({
                'date': date,
                'type': 'BUY',
                'price': price,
                'shares': shares
            })
        
        elif signal == 'SELL' and position == 1:
            capital = shares * price
            pnl = (price - entry_price) * shares
            pnl_pct = ((price - entry_price) / entry_price) * 100
            trades.append({
                'date': date,
                'type': 'SELL',
                'price': price,
                'shares': shares,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
            shares = 0
            position = 0
        
        current_equity = capital + (shares * price)
        equity_curve.append({
            'Date': date,
            'Equity': current_equity,
            'Position': position,
            'Price': price
        })
    
    equity_df = pd.DataFrame(equity_curve)
    
    stats = calculate_backtest_stats(equity_df, trades, initial_capital)
    
    return {
        'equity_curve': equity_df,
        'trades': trades,
        'stats': stats
    }


def calculate_backtest_stats(equity_df: pd.DataFrame, trades: list, 
                              initial_capital: float) -> dict:
    """
    Calculate backtest performance statistics.
    """
    if equity_df.empty:
        return {}
    
    final_equity = equity_df['Equity'].iloc[-1]
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    
    equity_df['Peak'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak'] * 100
    max_drawdown = equity_df['Drawdown'].min()
    
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
    total_closed_trades = len([t for t in trades if 'pnl' in t])
    
    win_rate = (len(winning_trades) / total_closed_trades * 100) if total_closed_trades > 0 else 0
    
    equity_df['Returns'] = equity_df['Equity'].pct_change()
    avg_return = equity_df['Returns'].mean()
    std_return = equity_df['Returns'].std()
    sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    
    return {
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'closed_trades': total_closed_trades,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }


def get_buy_and_hold_benchmark(df: pd.DataFrame, initial_capital: float = 10000) -> pd.DataFrame:
    """
    Calculate buy and hold benchmark for comparison.
    """
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    initial_price = df['Close'].iloc[0]
    shares = initial_capital / initial_price
    
    df['Benchmark'] = shares * df['Close']
    
    return df[['Date', 'Benchmark']]
