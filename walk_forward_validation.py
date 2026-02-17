"""
TRUE Walk-Forward Validation
=============================
3-block temporal split with NO data leakage:

  TRAIN:      Jan 2023 → Dec 2023   (retrain model from scratch)
  VALIDATE:   Jan 2024 → Jun 2024   (tune threshold ONCE)
  FINAL TEST: Jul 2024 → Feb 2026   (single run, never re-optimized)

Uses same TP/SL/time_limit as production decision model,
same realistic execution rules as backtesting_engine.py.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Base directory (where this script lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Date splits per coin ──────────────────────────────────────
SPLITS = {
    'BTC_USDT': {
        'train': ('2023-01-01', '2023-12-31'),
        'validate': ('2024-01-01', '2024-06-30'),
        'test': ('2024-07-01', '2026-02-28'),
    },
    'ETH_USDT': {
        'train': ('2023-01-01', '2023-12-31'),
        'validate': ('2024-01-01', '2024-06-30'),
        'test': ('2024-07-01', '2026-02-28'),
    },
    'SOL_USDT': {
        'train': ('2023-01-01', '2023-12-31'),
        'validate': ('2024-01-01', '2024-06-30'),
        'test': ('2024-07-01', '2026-02-28'),
    },
    'PEPE_USDT': {
        # PEPE only has data from Apr 2024
        'train': ('2024-04-08', '2024-12-31'),
        'validate': ('2025-01-01', '2025-04-30'),
        'test': ('2025-05-01', '2026-02-28'),
    },
}

# ── Trading parameters (same as production) ──────────────────
TP_PCT = 0.05       # 5% take profit
SL_PCT = 0.03       # 3% stop loss
TIME_LIMIT = 48     # hours
POSITION_SIZE = 0.30
INITIAL_CAPITAL = 10000

# Costs (Binance spot)
FEE = 0.0006
SLIPPAGE = 0.0003
SPREAD = 0.0002
ROUND_TRIP_COST = (FEE + SLIPPAGE + SPREAD) * 2  # ~0.22%

# Thresholds to test during validation
THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55]


def load_data(coin):
    """Load multi-timeframe feature CSV."""
    path = os.path.join(BASE_DIR, f"data/{coin}_multi_tf_features.csv")
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_feature_cols(df):
    """Get feature columns (exclude OHLCV, targets, meta)."""
    drop = ['timestamp', 'target_return', 'target_direction',
            'open', 'high', 'low', 'close', 'volume', 'decision_label']
    return [c for c in df.columns if c not in drop]


def create_decision_labels(df):
    """
    Create WIN/LOSS/NO_TRADE labels using TP/SL race.
    Same logic as train_decision_model.py.
    """
    labels = []
    for i in range(len(df) - TIME_LIMIT):
        entry_price = df.iloc[i]['close']
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)

        future = df.iloc[i + 1: i + 1 + TIME_LIMIT]

        tp_time = TIME_LIMIT + 1
        sl_time = TIME_LIMIT + 1

        for j, (_, row) in enumerate(future.iterrows()):
            if row['high'] >= tp_price and tp_time > TIME_LIMIT:
                tp_time = j
            if row['low'] <= sl_price and sl_time > TIME_LIMIT:
                sl_time = j

        if tp_time <= TIME_LIMIT and sl_time <= TIME_LIMIT:
            labels.append('WIN' if tp_time < sl_time else 'LOSS')
        elif tp_time <= TIME_LIMIT:
            labels.append('WIN')
        elif sl_time <= TIME_LIMIT:
            labels.append('LOSS')
        else:
            labels.append('NO_TRADE')

    labels.extend([np.nan] * TIME_LIMIT)
    return labels


def train_model(train_df, feature_cols):
    """Train RandomForest on the TRAIN block only."""
    X = train_df[feature_cols].fillna(0)
    y = train_df['decision_label']

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=30,
        min_samples_split=50,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X, y)

    classes = list(model.classes_)
    train_acc = (model.predict(X) == y).mean()

    return model, classes, train_acc


def simulate_trades(model, df, feature_cols, classes, threshold):
    """
    Realistic trade simulation (same rules as backtesting_engine.py):
    - Entry at next candle's OPEN
    - 1-candle execution delay
    - Worst-case: both TP+SL hit → SL wins
    - SL checked before TP
    - Costs as flat round-trip deduction
    """
    win_idx = classes.index('WIN')
    loss_idx = classes.index('LOSS')

    X = df[feature_cols].fillna(0)
    probas = model.predict_proba(X)
    # Normalize (sklearn version mismatch safety)
    row_sums = probas.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probas = probas / row_sums

    df = df.copy()
    df['win_prob'] = probas[:, win_idx]
    df['loss_prob'] = probas[:, loss_idx]

    capital = INITIAL_CAPITAL
    trades = []
    equity = [capital]

    in_position = False
    pending_entry = False
    entry_exec_idx = None
    entry_price = None
    position_capital = 0
    cooldown = 0
    signal_win_prob = 0
    tp_price = 0
    sl_price = 0

    for i in range(1, len(df) - TIME_LIMIT):
        row = df.iloc[i]

        if cooldown > 0:
            cooldown -= 1

        # Execute pending entry at this candle's OPEN
        if pending_entry:
            entry_exec_idx = i
            entry_price = row['open']
            position_capital = capital * POSITION_SIZE
            tp_price = entry_price * (1 + TP_PCT)
            sl_price = entry_price * (1 - SL_PCT)
            in_position = True
            pending_entry = False

        # Check exit (only after entry candle)
        if in_position and i > entry_exec_idx:
            hours_held = i - entry_exec_idx
            high = row['high']
            low = row['low']

            exit_reason = None
            exit_price = None

            tp_hit = high >= tp_price
            sl_hit = low <= sl_price

            if tp_hit and sl_hit:
                exit_reason = 'SL'
                exit_price = sl_price
            elif sl_hit:
                exit_reason = 'SL'
                exit_price = sl_price
            elif tp_hit:
                exit_reason = 'TP'
                exit_price = tp_price
            elif hours_held >= TIME_LIMIT:
                exit_reason = 'TIME'
                exit_price = row['close']

            if exit_reason:
                gross_pnl_pct = (exit_price - entry_price) / entry_price * 100
                net_pnl_pct = gross_pnl_pct - (ROUND_TRIP_COST * 100)
                pnl_usd = position_capital * (net_pnl_pct / 100)
                capital += pnl_usd

                trades.append({
                    'entry_time': df.iloc[entry_exec_idx]['timestamp'].isoformat(),
                    'exit_time': row['timestamp'].isoformat(),
                    'entry_price': round(entry_price, 8),
                    'exit_price': round(exit_price, 8),
                    'win_prob': round(signal_win_prob * 100, 1),
                    'net_pnl_pct': round(net_pnl_pct, 2),
                    'pnl_usd': round(pnl_usd, 2),
                    'exit_reason': exit_reason,
                    'hours_held': hours_held,
                    'result': 'WIN' if net_pnl_pct > 0 else 'LOSS',
                })

                in_position = False
                cooldown = 3

        # Generate signal → pending entry for next candle
        if not in_position and not pending_entry and cooldown == 0:
            win_prob = df.iloc[i]['win_prob']
            loss_prob = df.iloc[i]['loss_prob']
            if win_prob >= threshold and loss_prob < 0.40:
                pending_entry = True
                signal_win_prob = win_prob

        equity.append(capital)

    return trades, equity, capital


def calculate_metrics(trades, equity, final_capital):
    """Calculate backtest metrics from trade list."""
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
            'sharpe_ratio': 0, 'max_drawdown_pct': 0,
            'total_return_pct': 0, 'final_capital': INITIAL_CAPITAL,
        }

    total = len(trades)
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    win_rate = wins / total * 100

    gross_profit = sum(t['pnl_usd'] for t in trades if t['pnl_usd'] > 0)
    gross_loss = abs(sum(t['pnl_usd'] for t in trades if t['pnl_usd'] < 0))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.99

    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Sharpe from daily equity
    eq = pd.Series(equity)
    daily_eq = eq.iloc[::24]
    if len(daily_eq) > 2:
        daily_ret = daily_eq.pct_change().dropna()
        if daily_ret.std() > 0:
            sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365)
            sharpe = round(float(sharpe), 2)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    eq_series = pd.Series(equity)
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak * 100
    max_dd = round(float(dd.min()), 2)

    # Exit reason counts
    from collections import Counter
    reasons = Counter(t['exit_reason'] for t in trades)

    return {
        'total_trades': total,
        'winning_trades': wins,
        'losing_trades': total - wins,
        'win_rate': round(win_rate, 1),
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd,
        'total_return_pct': round(total_return, 2),
        'final_capital': round(final_capital, 2),
        'initial_capital': INITIAL_CAPITAL,
        'avg_win_pct': round(np.mean([t['net_pnl_pct'] for t in trades if t['result'] == 'WIN']), 2) if wins > 0 else 0,
        'avg_loss_pct': round(np.mean([t['net_pnl_pct'] for t in trades if t['result'] == 'LOSS']), 2) if (total - wins) > 0 else 0,
        'tp_exits': reasons.get('TP', 0),
        'sl_exits': reasons.get('SL', 0),
        'time_exits': reasons.get('TIME', 0),
    }


def run_walk_forward(coin):
    """
    Full 3-block walk-forward validation for one coin.
    Returns dict with train_stats, validation_results, final_test results.
    """
    splits = SPLITS[coin]
    print(f"\n{'='*60}")
    print(f"  {coin} — Walk-Forward Validation")
    print(f"{'='*60}")

    # ── Load data ─────────────────────────────────────────────
    df = load_data(coin)
    print(f"  Data: {len(df):,} rows ({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")

    # ── Create decision labels on FULL dataset ────────────────
    print(f"  Creating decision labels (TP={TP_PCT*100}% SL={SL_PCT*100}% T={TIME_LIMIT}h)...")
    df['decision_label'] = create_decision_labels(df)
    df = df.dropna(subset=['decision_label'])

    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}")

    # ── Split into 3 blocks ───────────────────────────────────
    train_df = df[(df['timestamp'] >= splits['train'][0]) &
                  (df['timestamp'] <= splits['train'][1])].copy()
    val_df = df[(df['timestamp'] >= splits['validate'][0]) &
                (df['timestamp'] <= splits['validate'][1])].copy()
    test_df = df[(df['timestamp'] >= splits['test'][0]) &
                 (df['timestamp'] <= splits['test'][1])].copy()

    print(f"\n  TRAIN:    {splits['train'][0]} → {splits['train'][1]}  ({len(train_df):,} rows)")
    print(f"  VALIDATE: {splits['validate'][0]} → {splits['validate'][1]}  ({len(val_df):,} rows)")
    print(f"  TEST:     {splits['test'][0]} → {splits['test'][1]}  ({len(test_df):,} rows)")

    if len(train_df) < 500:
        print(f"  ⚠️  Insufficient training data ({len(train_df)} rows). Skipping.")
        return {'coin': coin, 'error': 'Insufficient training data'}

    # Label distribution in train set
    dist = train_df['decision_label'].value_counts(normalize=True) * 100
    print(f"\n  Train label distribution:")
    for label in ['WIN', 'NO_TRADE', 'LOSS']:
        if label in dist.index:
            print(f"    {label}: {dist[label]:.1f}%")

    # ── Step 1: Train model on TRAIN block only ──────────────
    print(f"\n  Training RandomForest on TRAIN block...")
    model, classes, train_acc = train_model(train_df, feature_cols)
    print(f"  Train accuracy: {train_acc*100:.1f}%")

    # Save walk-forward model separately
    model_dir = os.path.join(BASE_DIR, f"models/{coin}")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/wf_decision_model.pkl")
    print(f"  Saved: {model_dir}/wf_decision_model.pkl")

    # ── Step 2: Tune threshold on VALIDATION block ───────────
    print(f"\n  Tuning threshold on VALIDATION block...")
    val_results = []
    for thresh in THRESHOLDS:
        trades, equity, final_cap = simulate_trades(
            model, val_df, feature_cols, classes, thresh
        )
        metrics = calculate_metrics(trades, equity, final_cap)
        val_results.append({
            'threshold': thresh,
            'trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe_ratio'],
            'return_pct': metrics['total_return_pct'],
            'profit_factor': metrics['profit_factor'],
            'max_dd': metrics['max_drawdown_pct'],
        })
        marker = ""
        print(f"    threshold={thresh:.2f}: {metrics['total_trades']:>3} trades, "
              f"WR={metrics['win_rate']:>5.1f}%, "
              f"Sharpe={metrics['sharpe_ratio']:>5.2f}, "
              f"Return={metrics['total_return_pct']:>+6.2f}%, "
              f"PF={metrics['profit_factor']:>5.2f}")

    # Pick best by Sharpe (must have ≥5 trades)
    valid = [r for r in val_results if r['trades'] >= 5]
    if not valid:
        print(f"  ⚠️  No threshold produced ≥5 trades on validation. Skipping.")
        return {'coin': coin, 'error': 'No valid threshold found on validation set'}

    best = max(valid, key=lambda x: x['sharpe'])
    chosen_threshold = best['threshold']
    print(f"\n  ✅ Chosen threshold: {chosen_threshold} (Sharpe={best['sharpe']:.2f})")

    # ── Step 3: Final test — SINGLE RUN, NO RE-OPTIMIZATION ──
    print(f"\n  Running FINAL TEST (threshold={chosen_threshold})...")
    test_trades, test_equity, test_final = simulate_trades(
        model, test_df, feature_cols, classes, chosen_threshold
    )
    test_metrics = calculate_metrics(test_trades, test_equity, test_final)

    print(f"\n  {'─'*50}")
    print(f"  FINAL TEST RESULTS — {coin}")
    print(f"  {'─'*50}")
    print(f"  Total trades:    {test_metrics['total_trades']}")
    print(f"  Win rate:        {test_metrics['win_rate']:.1f}%")
    print(f"  Profit factor:   {test_metrics['profit_factor']:.2f}")
    print(f"  Sharpe ratio:    {test_metrics['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:    {test_metrics['max_drawdown_pct']:.2f}%")
    print(f"  Total return:    {test_metrics['total_return_pct']:+.2f}%")
    print(f"  Final capital:   ${test_metrics['final_capital']:,.2f}")
    print(f"  {'─'*50}")

    # Verdict
    if test_metrics['total_trades'] < 20:
        verdict = 'INSUFFICIENT_DATA'
    elif test_metrics['sharpe_ratio'] >= 1.0 and test_metrics['win_rate'] >= 50:
        verdict = 'VIABLE'
    elif test_metrics['sharpe_ratio'] >= 0.5 and test_metrics['win_rate'] >= 45:
        verdict = 'MARGINAL'
    else:
        verdict = 'NOT_VIABLE'

    print(f"  Verdict: {verdict}")

    return {
        'coin': coin,
        'splits': splits,
        'train_stats': {
            'rows': len(train_df),
            'accuracy': round(train_acc * 100, 1),
            'label_distribution': {k: round(v, 1) for k, v in dist.to_dict().items()},
        },
        'validation': {
            'rows': len(val_df),
            'threshold_results': val_results,
            'chosen_threshold': chosen_threshold,
            'chosen_sharpe': best['sharpe'],
        },
        'test': {
            'rows': len(test_df),
            'threshold': chosen_threshold,
            'metrics': test_metrics,
            'trades': test_trades,
            'equity_curve': test_equity[::24],  # downsample to daily
        },
        'verdict': verdict,
    }


def run_all():
    """Run walk-forward validation for all coins."""
    print("\n" + "#" * 60)
    print("  TRUE WALK-FORWARD VALIDATION")
    print("  No data leakage. No re-optimization. Honest numbers.")
    print("#" * 60)

    results = {}
    for coin in ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']:
        results[coin] = run_walk_forward(coin)

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY — Walk-Forward Validation Results")
    print(f"{'='*70}")
    print(f"  {'Coin':<12} {'Trades':>7} {'WinRate':>8} {'PF':>7} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'Verdict'}")
    print(f"  {'-'*70}")

    for coin, r in results.items():
        if 'error' in r:
            print(f"  {coin:<12} ERROR: {r['error']}")
            continue
        m = r['test']['metrics']
        v = r['verdict']
        print(f"  {coin:<12} {m['total_trades']:>7} {m['win_rate']:>7.1f}% "
              f"{m['profit_factor']:>7.2f} {m['sharpe_ratio']:>7.2f} "
              f"{m['total_return_pct']:>+7.2f}% {m['max_drawdown_pct']:>6.2f}% {v}")

    print(f"\n  Cost model: fee={FEE*100:.2f}% + slip={SLIPPAGE*100:.2f}% + spread={SPREAD*100:.2f}% = {ROUND_TRIP_COST*100:.2f}% round-trip")
    print(f"  Execution: entry@next_open, worst-case SL, no same-candle exit")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    results = run_all()
