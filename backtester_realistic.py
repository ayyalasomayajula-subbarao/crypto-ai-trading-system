import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class RealisticBacktester:
    """
    Realistic backtester using OUT-OF-SAMPLE data only.
    
    Training data: First 80% (what model learned from)
    Test data: Last 20% (what we backtest on)
    """
    
    def __init__(self, coin, initial_capital=10000):
        self.coin = coin
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Realistic trading parameters
        self.tp_pct = 0.05      # 5% take profit
        self.sl_pct = 0.03      # 3% stop loss
        self.time_limit = 48    # 48 hours max hold
        self.position_size = 0.30  # 30% of capital (more conservative)
        self.fee_pct = 0.001    # 0.1% trading fee (Binance)
        
        self.trades = []
        self.load_model()
    
    def load_model(self):
        model_path = f"models/{self.coin}/decision_model.pkl"
        features_path = f"models/{self.coin}/decision_features.txt"
        
        self.model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            self.feature_cols = [line.strip() for line in f.readlines()]
        
        self.classes = list(self.model.classes_)
        self.win_idx = self.classes.index('WIN')
        self.loss_idx = self.classes.index('LOSS')
    
    def run_backtest(self, df, threshold=0.45):
        """Run backtest on TEST DATA ONLY (last 20%)"""
        
        print(f"\n{'='*65}")
        print(f"  REALISTIC BACKTEST: {self.coin}")
        print(f"  Threshold: WIN prob >= {threshold*100:.0f}%")
        print(f"{'='*65}")
        
        self.capital = self.initial_capital
        self.trades = []
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Split: Use only LAST 20% for backtesting (out-of-sample)
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        
        print(f"  Full data: {len(df):,} rows")
        print(f"  Training data (not used): {split_idx:,} rows (first 80%)")
        print(f"  Test data (backtesting): {len(test_df):,} rows (last 20%)")
        print(f"  Test period: {test_df['timestamp'].min().strftime('%Y-%m-%d')} to {test_df['timestamp'].max().strftime('%Y-%m-%d')}")
        
        # Get predictions on test data only
        X = test_df[self.feature_cols].fillna(0)
        probas = self.model.predict_proba(X)
        test_df['win_prob'] = probas[:, self.win_idx]
        test_df['loss_prob'] = probas[:, self.loss_idx]
        
        # Simulate trading
        in_position = False
        entry_idx = None
        entry_price = None
        position_capital = 0
        cooldown = 0  # Prevent overtrading
        
        equity_curve = [self.initial_capital]
        
        for i in range(len(test_df) - self.time_limit):
            current_price = test_df.iloc[i]['close']
            win_prob = test_df.iloc[i]['win_prob']
            loss_prob = test_df.iloc[i]['loss_prob']
            
            if cooldown > 0:
                cooldown -= 1
            
            # Check for exit if in position
            if in_position:
                hours_held = i - entry_idx
                high = test_df.iloc[i]['high']
                low = test_df.iloc[i]['low']
                
                tp_price = entry_price * (1 + self.tp_pct)
                sl_price = entry_price * (1 - self.sl_pct)
                
                exit_reason = None
                exit_price = None
                
                if high >= tp_price:
                    exit_reason = 'TP'
                    exit_price = tp_price
                elif low <= sl_price:
                    exit_reason = 'SL'
                    exit_price = sl_price
                elif hours_held >= self.time_limit:
                    exit_reason = 'TIME'
                    exit_price = current_price
                
                if exit_reason:
                    # Calculate P&L with fees
                    gross_pnl_pct = (exit_price - entry_price) / entry_price
                    fees = self.fee_pct * 2  # Entry + exit fee
                    net_pnl_pct = gross_pnl_pct - fees
                    
                    pnl_usd = position_capital * net_pnl_pct
                    self.capital += pnl_usd
                    
                    self.trades.append({
                        'entry_time': test_df.iloc[entry_idx]['timestamp'],
                        'exit_time': test_df.iloc[i]['timestamp'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'win_prob_at_entry': test_df.iloc[entry_idx]['win_prob'],
                        'gross_pnl_pct': gross_pnl_pct * 100,
                        'net_pnl_pct': net_pnl_pct * 100,
                        'pnl_usd': pnl_usd,
                        'exit_reason': exit_reason,
                        'hours_held': hours_held,
                        'result': 'WIN' if net_pnl_pct > 0 else 'LOSS'
                    })
                    
                    in_position = False
                    cooldown = 6  # Wait 6 hours before next trade
            
            # Check for entry
            if not in_position and cooldown == 0:
                # Entry conditions
                if win_prob >= threshold and loss_prob < 0.40:
                    in_position = True
                    entry_idx = i
                    entry_price = current_price
                    position_capital = self.capital * self.position_size
            
            equity_curve.append(self.capital)
        
        return self.calculate_results(equity_curve)
    
    def calculate_results(self, equity_curve):
        """Calculate realistic performance metrics"""
        
        if len(self.trades) == 0:
            print("\n  ⚠️ No trades executed!")
            return None
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['result'] == 'WIN'])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades * 100
        
        # Returns
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Average P&L
        avg_win = trades_df[trades_df['result'] == 'WIN']['net_pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['result'] == 'LOSS']['net_pnl_pct'].mean() if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        equity_series = pd.Series(equity_curve)
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # By exit reason
        tp_trades = len(trades_df[trades_df['exit_reason'] == 'TP'])
        sl_trades = len(trades_df[trades_df['exit_reason'] == 'SL'])
        time_trades = len(trades_df[trades_df['exit_reason'] == 'TIME'])
        
        # Display results
        print(f"\n{'━'*65}")
        print(f"  📊 REALISTIC BACKTEST RESULTS: {self.coin}")
        print(f"{'━'*65}")
        
        print(f"\n  💰 CAPITAL:")
        print(f"     Starting:        ${self.initial_capital:,.2f}")
        print(f"     Final:           ${self.capital:,.2f}")
        print(f"     Total Return:    {total_return:+.2f}%")
        
        print(f"\n  📈 TRADES:")
        print(f"     Total:           {total_trades}")
        print(f"     Winning:         {winning_trades} ({win_rate:.1f}%)")
        print(f"     Losing:          {losing_trades}")
        print(f"     Avg Win:         +{avg_win:.2f}%")
        print(f"     Avg Loss:        {avg_loss:.2f}%")
        
        print(f"\n  🎯 EXIT REASONS:")
        print(f"     Take Profit:     {tp_trades} ({tp_trades/total_trades*100:.1f}%)")
        print(f"     Stop Loss:       {sl_trades} ({sl_trades/total_trades*100:.1f}%)")
        print(f"     Time Stop:       {time_trades} ({time_trades/total_trades*100:.1f}%)")
        
        print(f"\n  📉 RISK:")
        print(f"     Max Drawdown:    {max_drawdown:.2f}%")
        print(f"     Profit Factor:   {profit_factor:.2f}")
        
        # Annualized return estimate
        test_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
        if test_days > 0:
            annual_return = (1 + total_return/100) ** (365/test_days) - 1
            print(f"\n  📅 ANNUALIZED:")
            print(f"     Test Period:     {test_days} days")
            print(f"     Annual Return:   {annual_return*100:+.2f}%")
        
        print(f"\n  {'━'*50}")
        if total_return > 50:
            print(f"  🏆 EXCELLENT - Strategy is highly profitable!")
        elif total_return > 20:
            print(f"  ✅ GOOD - Strategy beats most investments")
        elif total_return > 0:
            print(f"  👍 PROFITABLE - Positive returns")
        else:
            print(f"  ⚠️ UNPROFITABLE - Needs adjustment")
        print(f"  {'━'*50}")
        
        return {
            'coin': self.coin,
            'initial_capital': self.initial_capital,
            'final_capital': round(self.capital, 2),
            'total_return_pct': round(total_return, 2),
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'tp_exits': tp_trades,
            'sl_exits': sl_trades,
            'time_exits': time_trades
        }


def run_all_backtests():
    """Run realistic backtest for all coins"""
    
    print("\n" + "#"*65)
    print("  💰 REALISTIC BACKTEST - OUT-OF-SAMPLE DATA")
    print("  (Testing on data the model has NEVER seen)")
    print("#"*65)
    
    coins = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']
    all_results = {}
    
    for coin in coins:
        data_path = f"data/{coin}_multi_tf_features.csv"
        if not os.path.exists(data_path):
            print(f"\n  ⚠️ Data not found for {coin}")
            continue
        
        df = pd.read_csv(data_path)
        
        backtester = RealisticBacktester(coin, initial_capital=10000)
        results = backtester.run_backtest(df, threshold=0.45)
        
        if results:
            all_results[coin] = results
    
    # Summary
    print("\n" + "="*65)
    print("  📊 FINAL SUMMARY - ALL COINS")
    print("="*65)
    
    print(f"\n  {'Coin':<12} {'Return':<12} {'Win Rate':<10} {'Trades':<8} {'Max DD':<10} {'PF':<8}")
    print("  " + "-"*60)
    
    total_return = 0
    for coin, r in all_results.items():
        print(f"  {coin:<12} {r['total_return_pct']:>+8.1f}%   {r['win_rate']:>6.1f}%    {r['total_trades']:>5}    {r['max_drawdown_pct']:>7.1f}%   {r['profit_factor']:>5.2f}")
        total_return += r['total_return_pct']
    
    print("  " + "-"*60)
    avg_return = total_return / len(all_results) if all_results else 0
    print(f"  {'AVERAGE':<12} {avg_return:>+8.1f}%")
    
    # Save results
    os.makedirs('backtest_results', exist_ok=True)
    summary_df = pd.DataFrame(list(all_results.values()))
    summary_df.to_csv('backtest_results/realistic_summary.csv', index=False)
    print(f"\n  💾 Saved: backtest_results/realistic_summary.csv")
    
    print("\n" + "#"*65)
    print("  🎉 REALISTIC BACKTEST COMPLETE!")
    print("#"*65)
    
    return all_results


if __name__ == "__main__":
    run_all_backtests()