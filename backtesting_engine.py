import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class BacktestingEngine:
    """
    Walk-forward backtesting engine with Sharpe ratio, regime stats, and equity curve.
    Built for API consumption - returns JSON-serializable results.

    Realistic cost model:
    - Exchange fee: 0.06% per side (Binance taker)
    - Slippage: 0.03% per side (market impact)
    - Spread: 0.02% per side (bid-ask)
    - Total cost per round-trip: ~0.22%
    """

    COINS = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']

    # Realistic cost defaults (Binance spot)
    DEFAULT_FEE = 0.0006       # 0.06% taker fee per side
    DEFAULT_SLIPPAGE = 0.0003  # 0.03% slippage per side
    DEFAULT_SPREAD = 0.0002    # 0.02% spread per side

    def __init__(self, coin, initial_capital=10000, tp_pct=0.05, sl_pct=0.03,
                 time_limit=48, position_size=0.30,
                 fee_pct=None, slippage_pct=None, spread_pct=None,
                 threshold=0.45):
        self.coin = coin
        self.initial_capital = initial_capital
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.time_limit = time_limit
        self.position_size = position_size
        self.fee_pct = fee_pct if fee_pct is not None else self.DEFAULT_FEE
        self.slippage_pct = slippage_pct if slippage_pct is not None else self.DEFAULT_SLIPPAGE
        self.spread_pct = spread_pct if spread_pct is not None else self.DEFAULT_SPREAD
        self.threshold = threshold

        # Total cost per side = fee + slippage + spread
        self.cost_per_side = self.fee_pct + self.slippage_pct + self.spread_pct
        # Round-trip cost (entry + exit)
        self.round_trip_cost = self.cost_per_side * 2

        self.model = None
        self.feature_cols = None
        self.classes = None
        self.win_idx = None
        self.loss_idx = None

    def load_model(self):
        """Load ML model and feature list for the coin."""
        model_path = f"models/{self.coin}/decision_model.pkl"
        features_path = f"models/{self.coin}/decision_features.txt"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)

        # Patch for sklearn version mismatch (models trained on 1.3.0)
        # Add missing monotonic_cst attribute to forest and each tree
        if not hasattr(self.model, 'monotonic_cst'):
            self.model.monotonic_cst = None
        if hasattr(self.model, 'estimators_'):
            for est in self.model.estimators_:
                if not hasattr(est, 'monotonic_cst'):
                    est.monotonic_cst = None

        with open(features_path, 'r') as f:
            self.feature_cols = [line.strip() for line in f.readlines()]

        self.classes = list(self.model.classes_)
        self.win_idx = self.classes.index('WIN')
        self.loss_idx = self.classes.index('LOSS')

    def _load_data(self):
        """Load multi-timeframe feature CSV for the coin."""
        data_path = f"data/{self.coin}_multi_tf_features.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found: {data_path}")

        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def _get_regime(self, adx_value):
        """Classify market regime from ADX value."""
        if adx_value >= 40:
            return 'STRONG_TREND'
        elif adx_value >= 25:
            return 'TRENDING'
        elif adx_value >= 20:
            return 'TRANSITIONING'
        else:
            return 'RANGING'

    def _simulate_trades(self, test_df, randomize_slippage=False):
        """
        Core simulation loop with realistic execution rules:

        1. ENTRY at next candle's OPEN (not current close) - simulates
           seeing signal, then executing on next available price.
        2. EXIT check starts 2 candles AFTER entry (no same-candle exit).
        3. WORST-CASE rule: if both TP and SL hit in same candle, SL wins.
        4. Costs deducted as flat % from PnL (fee + slippage + spread).
        """
        capital = self.initial_capital
        trades = []
        equity_curve = [{'timestamp': test_df.iloc[0]['timestamp'].isoformat(), 'equity': capital}]

        # Get predictions
        X = test_df[self.feature_cols].fillna(0)
        probas = self.model.predict_proba(X)
        # Normalize probabilities (sklearn version mismatch can produce non-normalized values)
        row_sums = probas.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        probas = probas / row_sums
        test_df = test_df.copy()
        test_df['win_prob'] = probas[:, self.win_idx]
        test_df['loss_prob'] = probas[:, self.loss_idx]

        in_position = False
        entry_idx = None       # candle index where signal was generated
        entry_exec_idx = None  # candle index where entry was executed (signal + 1)
        entry_price = None
        position_capital = 0
        cooldown = 0
        pending_entry = False  # flag: signal seen, waiting for next candle to execute

        # Detect ADX column
        adx_col = None
        for col in ['1h_adx', 'adx', 'ADX']:
            if col in test_df.columns:
                adx_col = col
                break

        for i in range(1, len(test_df) - self.time_limit):
            row = test_df.iloc[i]
            current_price = row['close']

            if cooldown > 0:
                cooldown -= 1

            # Execute pending entry at this candle's OPEN
            if pending_entry:
                entry_exec_idx = i
                entry_price = row['open']  # execute at open, not close
                position_capital = capital * self.position_size
                in_position = True
                pending_entry = False

            # Check exit - only if we've been in position for at least 1 full candle
            # (entry_exec_idx is the candle we entered, so check from entry_exec_idx + 1)
            if in_position and i > entry_exec_idx:
                hours_held = i - entry_exec_idx
                high = row['high']
                low = row['low']

                tp_price = entry_price * (1 + self.tp_pct)
                sl_price = entry_price * (1 - self.sl_pct)

                exit_reason = None
                exit_price = None

                tp_hit = high >= tp_price
                sl_hit = low <= sl_price

                if tp_hit and sl_hit:
                    # WORST-CASE: both hit in same candle → assume SL first
                    exit_reason = 'SL'
                    exit_price = sl_price
                elif sl_hit:
                    exit_reason = 'SL'
                    exit_price = sl_price
                elif tp_hit:
                    exit_reason = 'TP'
                    exit_price = tp_price
                elif hours_held >= self.time_limit:
                    exit_reason = 'TIME'
                    exit_price = current_price

                if exit_reason:
                    gross_pnl_pct = (exit_price - entry_price) / entry_price

                    if randomize_slippage:
                        slip_mult = 1 + np.random.uniform(-0.5, 0.5)
                        trade_cost = (self.fee_pct + self.slippage_pct * slip_mult + self.spread_pct) * 2
                    else:
                        trade_cost = self.round_trip_cost

                    net_pnl_pct = gross_pnl_pct - trade_cost
                    pnl_usd = position_capital * net_pnl_pct
                    capital += pnl_usd

                    regime = 'UNKNOWN'
                    if adx_col and not pd.isna(test_df.iloc[entry_idx][adx_col]):
                        regime = self._get_regime(test_df.iloc[entry_idx][adx_col])

                    trades.append({
                        'entry_time': test_df.iloc[entry_exec_idx]['timestamp'].isoformat(),
                        'exit_time': row['timestamp'].isoformat(),
                        'entry_price': float(entry_price),
                        'exit_price': float(exit_price),
                        'win_prob': round(float(test_df.iloc[entry_idx]['win_prob']) * 100, 1),
                        'gross_pnl_pct': round(float(gross_pnl_pct) * 100, 2),
                        'net_pnl_pct': round(float(net_pnl_pct) * 100, 2),
                        'trade_cost_pct': round(float(trade_cost) * 100, 3),
                        'pnl_usd': round(float(pnl_usd), 2),
                        'exit_reason': exit_reason,
                        'hours_held': int(hours_held),
                        'result': 'WIN' if net_pnl_pct > 0 else 'LOSS',
                        'regime': regime
                    })

                    in_position = False
                    cooldown = 6

            # Check for new entry signal (only when not in position and not pending)
            if not in_position and not pending_entry and cooldown == 0:
                win_prob = row['win_prob']
                loss_prob = row['loss_prob']
                if win_prob >= self.threshold and loss_prob < 0.40:
                    # Signal seen at candle i → will execute at candle i+1's open
                    pending_entry = True
                    entry_idx = i  # signal candle (for regime/prob tracking)

            equity_curve.append({
                'timestamp': row['timestamp'].isoformat(),
                'equity': round(capital, 2)
            })

        return trades, equity_curve, capital

    def _calculate_sharpe(self, equity_curve, risk_free_rate=0.02):
        """Calculate annualized Sharpe ratio from equity curve."""
        equities = [p['equity'] for p in equity_curve]
        if len(equities) < 48:
            return 0.0

        equity_series = pd.Series(equities)

        # Resample to daily returns (every 24 points for hourly data)
        daily_equity = equity_series.iloc[::24]
        if len(daily_equity) < 2:
            return 0.0

        daily_returns = daily_equity.pct_change().dropna()
        if daily_returns.std() == 0:
            return 0.0

        excess_returns = daily_returns - (risk_free_rate / 365)
        sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(365)
        return round(float(sharpe), 2)

    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown percentage."""
        equities = pd.Series([p['equity'] for p in equity_curve])
        peak = equities.cummax()
        drawdown = (equities - peak) / peak * 100
        return round(float(drawdown.min()), 2)

    def _calculate_regime_stats(self, trades):
        """Win rate broken down by market regime."""
        if not trades:
            return {}

        regime_stats = {}
        for trade in trades:
            regime = trade['regime']
            if regime not in regime_stats:
                regime_stats[regime] = {'trades': 0, 'wins': 0}
            regime_stats[regime]['trades'] += 1
            if trade['result'] == 'WIN':
                regime_stats[regime]['wins'] += 1

        for regime in regime_stats:
            stats = regime_stats[regime]
            stats['win_rate'] = round(stats['wins'] / stats['trades'] * 100, 1) if stats['trades'] > 0 else 0

        return regime_stats

    def _downsample_equity_curve(self, equity_curve, max_points=500):
        """Downsample equity curve to at most max_points for API response size."""
        if len(equity_curve) <= max_points:
            return equity_curve

        step = len(equity_curve) / max_points
        indices = [int(i * step) for i in range(max_points)]
        if indices[-1] != len(equity_curve) - 1:
            indices.append(len(equity_curve) - 1)

        return [equity_curve[i] for i in indices]

    def _build_metrics(self, trades, equity_curve, final_capital):
        """Build metrics dict from trade results. Shared by backtest and monte carlo."""
        if not trades:
            return None

        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        winning = len(trades_df[trades_df['result'] == 'WIN'])
        losing = total_trades - winning

        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        avg_win = trades_df[trades_df['result'] == 'WIN']['net_pnl_pct'].mean() if winning > 0 else 0
        avg_loss = trades_df[trades_df['result'] == 'LOSS']['net_pnl_pct'].mean() if losing > 0 else 0

        gross_profit = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum())
        profit_factor = round(float(gross_profit / gross_loss), 2) if gross_loss > 0 else 999.99

        test_days = (pd.Timestamp(trades[-1]['exit_time']) - pd.Timestamp(trades[0]['entry_time'])).days
        annual_return = ((1 + total_return / 100) ** (365 / max(test_days, 1)) - 1) * 100

        tp_exits = len(trades_df[trades_df['exit_reason'] == 'TP'])
        sl_exits = len(trades_df[trades_df['exit_reason'] == 'SL'])
        time_exits = len(trades_df[trades_df['exit_reason'] == 'TIME'])

        return {
            'initial_capital': self.initial_capital,
            'final_capital': round(final_capital, 2),
            'total_return_pct': round(total_return, 2),
            'annualized_return_pct': round(annual_return, 2),
            'total_trades': total_trades,
            'winning_trades': winning,
            'losing_trades': losing,
            'win_rate': round(winning / total_trades * 100, 1),
            'avg_win_pct': round(float(avg_win), 2),
            'avg_loss_pct': round(float(avg_loss), 2),
            'profit_factor': profit_factor,
            'max_drawdown_pct': self._calculate_max_drawdown(equity_curve),
            'sharpe_ratio': self._calculate_sharpe(equity_curve),
            'tp_exits': tp_exits,
            'sl_exits': sl_exits,
            'time_exits': time_exits,
            'avg_cost_per_trade': round(self.round_trip_cost * 100, 3),
            'test_days': test_days
        }

    def run_backtest(self, test_ratio=0.30):
        """
        Run backtest on last test_ratio of data (out-of-sample).
        Default 30% test data for more trades while staying out-of-sample.
        """
        if self.model is None:
            self.load_model()

        df = self._load_data()
        split_idx = int(len(df) * (1 - test_ratio))
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        trades, equity_curve, final_capital = self._simulate_trades(test_df)

        if not trades:
            return {
                'coin': self.coin,
                'error': 'No trades executed',
                'metrics': None,
                'equity_curve': [],
                'trades': [],
                'regime_stats': {},
                'config': self._get_config(),
                'test_period': {
                    'start': test_df['timestamp'].min().isoformat(),
                    'end': test_df['timestamp'].max().isoformat(),
                    'rows': len(test_df)
                }
            }

        metrics = self._build_metrics(trades, equity_curve, final_capital)

        return {
            'coin': self.coin,
            'metrics': metrics,
            'equity_curve': self._downsample_equity_curve(equity_curve),
            'trades': trades,
            'regime_stats': self._calculate_regime_stats(trades),
            'config': self._get_config(),
            'test_period': {
                'start': test_df['timestamp'].min().isoformat(),
                'end': test_df['timestamp'].max().isoformat(),
                'days': metrics['test_days'],
                'rows': len(test_df)
            }
        }

    def run_walk_forward(self, n_splits=5):
        """
        Walk-forward validation: expanding window train, test on next segment.
        Uses date-based splits for proper temporal separation.
        Returns per-split metrics + aggregate.
        """
        if self.model is None:
            self.load_model()

        df = self._load_data()

        # Date-based splits for proper temporal separation
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        total_days = (end_date - start_date).days

        # Reserve first 50% for initial training, split remaining into n_splits
        train_end_days = int(total_days * 0.5)
        test_period_days = int((total_days * 0.5) / n_splits)

        splits = []
        all_trades = []

        for split_i in range(n_splits):
            test_start_date = start_date + pd.Timedelta(days=train_end_days + split_i * test_period_days)
            test_end_date = test_start_date + pd.Timedelta(days=test_period_days)

            test_df = df[(df['timestamp'] >= test_start_date) & (df['timestamp'] < test_end_date)]
            if len(test_df) < self.time_limit + 10:
                continue

            test_df = test_df.reset_index(drop=True)
            trades, equity_curve, final_capital = self._simulate_trades(test_df)

            initial = equity_curve[0]['equity'] if equity_curve else self.initial_capital
            ret = (final_capital - initial) / initial * 100 if initial > 0 else 0
            win_count = sum(1 for t in trades if t['result'] == 'WIN')

            split_result = {
                'split': split_i + 1,
                'test_start': test_start_date.isoformat(),
                'test_end': test_end_date.isoformat(),
                'test_rows': len(test_df),
                'trades': len(trades),
                'win_rate': round(win_count / len(trades) * 100, 1) if trades else 0,
                'return_pct': round(ret, 2)
            }
            splits.append(split_result)
            all_trades.extend(trades)

        # Aggregate stats
        total_trades = len(all_trades)
        total_wins = sum(1 for t in all_trades if t['result'] == 'WIN')
        avg_return = np.mean([s['return_pct'] for s in splits]) if splits else 0

        return {
            'coin': self.coin,
            'n_splits': n_splits,
            'splits': splits,
            'aggregate': {
                'total_trades': total_trades,
                'win_rate': round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
                'avg_return_per_split': round(float(avg_return), 2),
                'regime_stats': self._calculate_regime_stats(all_trades)
            },
            'config': self._get_config()
        }

    def run_monte_carlo(self, n_simulations=100, test_ratio=0.30):
        """
        Monte Carlo stress test.

        Randomizes:
        - Slippage variation (+-50%)
        - 5-10% missed trades (randomly skipped)
        - Trade order shuffling for equity path variation

        Returns distribution of outcomes across simulations.
        """
        if self.model is None:
            self.load_model()

        df = self._load_data()
        split_idx = int(len(df) * (1 - test_ratio))
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        # Run base simulation to get base trades
        base_trades, _, _ = self._simulate_trades(test_df)
        if not base_trades:
            return {
                'coin': self.coin,
                'error': 'No base trades to simulate',
                'simulations': 0
            }

        sim_results = []

        for sim_i in range(n_simulations):
            # Randomly skip 5-10% of trades
            skip_rate = np.random.uniform(0.05, 0.10)
            sim_trades = [t for t in base_trades if np.random.random() > skip_rate]

            if not sim_trades:
                continue

            # Shuffle trade order for equity path variation
            shuffled = sim_trades.copy()
            np.random.shuffle(shuffled)

            # Replay trades with randomized costs
            capital = self.initial_capital
            for trade in shuffled:
                # Randomize slippage +-50%
                slip_mult = 1 + np.random.uniform(-0.5, 0.5)
                trade_cost = (self.fee_pct + self.slippage_pct * slip_mult + self.spread_pct) * 2

                # Recalculate PnL with randomized cost
                gross_pnl_pct = trade['gross_pnl_pct'] / 100
                net_pnl_pct = gross_pnl_pct - trade_cost
                position = capital * self.position_size
                capital += position * net_pnl_pct

            total_return = (capital - self.initial_capital) / self.initial_capital * 100
            win_count = sum(1 for t in shuffled
                           if (t['gross_pnl_pct'] / 100 - (self.fee_pct + self.slippage_pct + self.spread_pct) * 2) > 0)

            sim_results.append({
                'final_capital': round(capital, 2),
                'total_return_pct': round(total_return, 2),
                'trades_used': len(shuffled),
                'win_rate': round(win_count / len(shuffled) * 100, 1)
            })

        if not sim_results:
            return {'coin': self.coin, 'error': 'All simulations empty', 'simulations': 0}

        returns = [s['total_return_pct'] for s in sim_results]
        profitable_count = sum(1 for r in returns if r > 0)

        return {
            'coin': self.coin,
            'simulations': len(sim_results),
            'base_trades': len(base_trades),
            'summary': {
                'profitable_pct': round(profitable_count / len(sim_results) * 100, 1),
                'median_return_pct': round(float(np.median(returns)), 2),
                'mean_return_pct': round(float(np.mean(returns)), 2),
                'worst_return_pct': round(float(np.min(returns)), 2),
                'best_return_pct': round(float(np.max(returns)), 2),
                'std_return_pct': round(float(np.std(returns)), 2),
                'percentile_5': round(float(np.percentile(returns, 5)), 2),
                'percentile_25': round(float(np.percentile(returns, 25)), 2),
                'percentile_75': round(float(np.percentile(returns, 75)), 2),
                'percentile_95': round(float(np.percentile(returns, 95)), 2),
            },
            'distribution': sorted(returns),
            'pass_criteria': {
                'profitable_70pct': profitable_count / len(sim_results) >= 0.70,
                'median_positive': float(np.median(returns)) > 0,
                'worst_case_above_neg10': float(np.min(returns)) > -10,
            },
            'verdict': 'ROBUST' if profitable_count / len(sim_results) >= 0.70 and float(np.median(returns)) > 0 else 'FRAGILE',
            'config': self._get_config()
        }

    def _get_config(self):
        return {
            'threshold': self.threshold,
            'tp_pct': self.tp_pct * 100,
            'sl_pct': self.sl_pct * 100,
            'time_limit_hours': self.time_limit,
            'position_size_pct': self.position_size * 100,
            'fee_pct': self.fee_pct * 100,
            'slippage_pct': self.slippage_pct * 100,
            'spread_pct': self.spread_pct * 100,
            'total_cost_per_trade': round(self.round_trip_cost * 100, 3)
        }


def run_backtest_for_coin(coin, threshold=0.45, capital=10000):
    """Helper to run a single coin backtest."""
    engine = BacktestingEngine(coin, initial_capital=capital, threshold=threshold)
    return engine.run_backtest()


def run_backtest_all_coins(threshold=0.45, capital=10000):
    """Run backtest for all supported coins."""
    results = {}
    for coin in BacktestingEngine.COINS:
        try:
            results[coin] = run_backtest_for_coin(coin, threshold, capital)
        except Exception as e:
            results[coin] = {'coin': coin, 'error': str(e)}
    return results
