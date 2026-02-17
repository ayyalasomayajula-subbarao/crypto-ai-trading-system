"""
Rolling Walk-Forward Robustness Validation
============================================
Tests strategy across multiple market regimes using expanding
training windows with yearly test periods.

PASS rule — a coin is robust if:
  - >= 3 different test years profitable
  - Profit factor >= 1.8
  - Max drawdown <= 20%

Reuses: create_decision_labels, train_model, simulate_trades,
        calculate_metrics from walk_forward_validation.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from walk_forward_validation import (
    create_decision_labels,
    train_model,
    simulate_trades,
    calculate_metrics,
    get_feature_cols,
    THRESHOLDS,
)


# ── Rolling windows per coin ─────────────────────────────────
# Adjusted for actual feature CSV start dates:
#   BTC/ETH: 2018-07-23, SOL: 2021-07-19, PEPE: 2024-04-08

ROLLING_WINDOWS = {
    'BTC_USDT': [
        {'train': ('2018-07-23', '2019-12-31'), 'test': ('2020-01-01', '2020-12-31'), 'label': '2020'},
        {'train': ('2018-07-23', '2020-12-31'), 'test': ('2021-01-01', '2021-12-31'), 'label': '2021'},
        {'train': ('2018-07-23', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31'), 'label': '2022'},
        {'train': ('2018-07-23', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31'), 'label': '2023'},
        {'train': ('2018-07-23', '2023-12-31'), 'test': ('2024-01-01', '2026-02-28'), 'label': '2024-26'},
    ],
    'ETH_USDT': [
        {'train': ('2018-07-23', '2019-12-31'), 'test': ('2020-01-01', '2020-12-31'), 'label': '2020'},
        {'train': ('2018-07-23', '2020-12-31'), 'test': ('2021-01-01', '2021-12-31'), 'label': '2021'},
        {'train': ('2018-07-23', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31'), 'label': '2022'},
        {'train': ('2018-07-23', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31'), 'label': '2023'},
        {'train': ('2018-07-23', '2023-12-31'), 'test': ('2024-01-01', '2026-02-28'), 'label': '2024-26'},
    ],
    'SOL_USDT': [
        {'train': ('2021-07-19', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31'), 'label': '2022'},
        {'train': ('2021-07-19', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31'), 'label': '2023'},
        {'train': ('2021-07-19', '2023-12-31'), 'test': ('2024-01-01', '2026-02-28'), 'label': '2024-26'},
    ],
    'PEPE_USDT': [
        {'train': ('2024-04-08', '2024-12-31'), 'test': ('2025-01-01', '2026-02-28'), 'label': '2025-26'},
    ],
}


def load_data(coin):
    """Load multi-timeframe feature CSV."""
    path = os.path.join(BASE_DIR, f"data/{coin}_multi_tf_features.csv")
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def pick_threshold(model, val_df, feature_cols, classes):
    """Pick best threshold by Sharpe on validation slice."""
    best_sharpe = -999
    best_thresh = 0.40  # default fallback

    for thresh in THRESHOLDS:
        trades, equity, final_cap = simulate_trades(
            model, val_df, feature_cols, classes, thresh
        )
        if len(trades) < 3:
            continue
        metrics = calculate_metrics(trades, equity, final_cap)
        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_thresh = thresh

    return best_thresh, best_sharpe


def run_single_window(coin, df, feature_cols, window):
    """Run one train→test window. Returns metrics dict."""
    train_start, train_end = window['train']
    test_start, test_end = window['test']
    label = window['label']

    # Split
    train_df = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)].copy()
    test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)].copy()

    if len(train_df) < 500:
        return {'label': label, 'error': f'Insufficient train data ({len(train_df)} rows)'}
    if len(test_df) < 200:
        return {'label': label, 'error': f'Insufficient test data ({len(test_df)} rows)'}

    # Create labels on train block
    train_df['decision_label'] = create_decision_labels(train_df)
    train_df = train_df.dropna(subset=['decision_label'])

    if len(train_df) < 300:
        return {'label': label, 'error': 'Insufficient labeled train data'}

    # Train
    model, classes, train_acc = train_model(train_df, feature_cols)

    # Pick threshold on last 20% of train as mini-validation
    val_split = int(len(train_df) * 0.8)
    val_slice = train_df.iloc[val_split:].copy()
    chosen_thresh, chosen_sharpe = pick_threshold(model, val_slice, feature_cols, classes)

    # Run test
    trades, equity, final_cap = simulate_trades(
        model, test_df, feature_cols, classes, chosen_thresh
    )
    metrics = calculate_metrics(trades, equity, final_cap)

    return {
        'label': label,
        'train_period': f"{train_start} → {train_end}",
        'test_period': f"{test_start} → {test_end}",
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'train_accuracy': round(float(train_acc) * 100, 1),
        'threshold': float(chosen_thresh),
        'metrics': metrics,
    }


def evaluate_robustness(results):
    """
    PASS rule:
    - >= 3 different test years profitable
    - Average PF >= 1.8
    - No single year DD > 20%
    """
    valid = [r for r in results if 'error' not in r and r['metrics']['total_trades'] >= 5]

    if len(valid) < 2:
        return 'INSUFFICIENT_WINDOWS', {
            'profitable_years': 0,
            'avg_pf': 0,
            'worst_dd': 0,
            'reason': f'Only {len(valid)} valid test windows',
        }

    profitable_years = sum(1 for r in valid if r['metrics']['total_return_pct'] > 0)
    avg_pf = np.mean([r['metrics']['profit_factor'] for r in valid])
    worst_dd = min(r['metrics']['max_drawdown_pct'] for r in valid)

    details = {
        'profitable_years': int(profitable_years),
        'total_years': len(valid),
        'avg_pf': round(float(avg_pf), 2),
        'worst_dd': round(float(worst_dd), 2),
        'pass_3_years': bool(profitable_years >= 3),
        'pass_pf': bool(float(avg_pf) >= 1.8),
        'pass_dd': bool(float(worst_dd) >= -20),
    }

    if profitable_years >= 3 and avg_pf >= 1.8 and worst_dd >= -20:
        return 'PASS', details
    else:
        reasons = []
        if profitable_years < 3:
            reasons.append(f'Only {profitable_years} profitable years (need 3+)')
        if avg_pf < 1.8:
            reasons.append(f'Avg PF {avg_pf:.2f} < 1.8')
        if worst_dd < -20:
            reasons.append(f'Worst DD {worst_dd:.1f}% exceeds -20%')
        details['reason'] = '; '.join(reasons)
        return 'FAIL', details


def run_coin_robustness(coin):
    """Run all rolling windows for one coin."""
    windows = ROLLING_WINDOWS[coin]

    print(f"\n{'='*60}")
    print(f"  {coin} — Rolling Robustness ({len(windows)} windows)")
    print(f"{'='*60}")

    df = load_data(coin)
    feature_cols = get_feature_cols(df)
    print(f"  Data: {len(df):,} rows ({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")

    results = []
    for i, window in enumerate(windows):
        print(f"\n  Window {i+1}/{len(windows)}: Train {window['train'][0]}→{window['train'][1]} | Test {window['label']}")
        result = run_single_window(coin, df, feature_cols, window)

        if 'error' in result:
            print(f"    ERROR: {result['error']}")
        else:
            m = result['metrics']
            print(f"    Threshold={result['threshold']} | {m['total_trades']} trades | "
                  f"WR={m['win_rate']:.1f}% | PF={m['profit_factor']:.2f} | "
                  f"Return={m['total_return_pct']:+.2f}% | DD={m['max_drawdown_pct']:.2f}%")

        results.append(result)

    verdict, details = evaluate_robustness(results)

    print(f"\n  {'─'*50}")
    print(f"  ROBUSTNESS VERDICT: {verdict}")
    for k, v in details.items():
        if k != 'reason':
            print(f"    {k}: {v}")
    if 'reason' in details:
        print(f"    Reason: {details['reason']}")
    print(f"  {'─'*50}")

    return {
        'coin': coin,
        'windows': results,
        'verdict': verdict,
        'robustness_details': details,
    }


def run_all():
    """Run rolling robustness for all coins."""
    print("\n" + "#" * 60)
    print("  ROLLING WALK-FORWARD ROBUSTNESS VALIDATION")
    print("  Multiple market regimes. Expanding windows. No cheating.")
    print("#" * 60)

    all_results = {}
    for coin in ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']:
        all_results[coin] = run_coin_robustness(coin)

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"  ROBUSTNESS SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"  {'Coin':<12} {'Test Year':<10} {'Trades':>7} {'WinRate':>8} {'PF':>7} {'Return':>8} {'MaxDD':>7} {'Sharpe':>7}")
    print(f"  {'-'*75}")

    for coin, data in all_results.items():
        for w in data['windows']:
            if 'error' in w:
                print(f"  {coin:<12} {w['label']:<10} {'ERROR: ' + w['error']}")
            else:
                m = w['metrics']
                print(f"  {coin:<12} {w['label']:<10} {m['total_trades']:>7} {m['win_rate']:>7.1f}% "
                      f"{m['profit_factor']:>7.2f} {m['total_return_pct']:>+7.2f}% "
                      f"{m['max_drawdown_pct']:>6.2f}% {m['sharpe_ratio']:>7.2f}")
        # Verdict row
        v = data['verdict']
        color = 'PASS' if v == 'PASS' else 'FAIL'
        print(f"  {coin:<12} {'VERDICT':<10} → {v}")
        print(f"  {'-'*75}")

    print(f"{'='*90}\n")

    return all_results


if __name__ == "__main__":
    results = run_all()
