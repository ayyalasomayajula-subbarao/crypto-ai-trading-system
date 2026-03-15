#!/usr/bin/env python3
"""
fix_labels.py — Recompute target_direction labels on existing feature CSVs.

Patches CSVs in-place (no exchange re-fetch needed).  Run time: <5s total.

Usage:
    python fix_labels.py              # all coins
    python fix_labels.py BTC SOL      # specific coins
"""

import sys
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Simple direction thresholds — % 24h return to qualify as UP / DOWN.
# Coin-specific, calibrated so UP and DOWN are each ~20-35% of bars.
# These are the PROVEN working labels (4 MARGINAL coins before Triple Barrier).
DIRECTION_THRESHOLDS = {
    'BTC_USDT':  1.5,
    'ETH_USDT':  1.5,
    'SOL_USDT':  3.0,   # higher beta — 1.5% is intraday noise for SOL
    'PEPE_USDT': 5.0,   # meme coin volatility
    'AVAX_USDT': 2.5,
    'BNB_USDT':  2.0,
    'LINK_USDT': 2.5,
}

ALL_COINS = list(DIRECTION_THRESHOLDS.keys())


def fix_coin(coin: str) -> None:
    csv_path = os.path.join(BASE_DIR, 'data', f'{coin}_multi_tf_features.csv')
    if not os.path.exists(csv_path):
        print(f"  [{coin}] ⚠️  CSV not found — skipping")
        return

    thresh = DIRECTION_THRESHOLDS[coin]
    print(f"\n[{coin}]  threshold=±{thresh}%")
    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"  {n:,} rows", flush=True)

    # Show old distribution
    if 'target_direction' in df.columns:
        old = df['target_direction'].value_counts(normalize=True) * 100
        print(f"  Old: UP={old.get('UP', 0):.1f}%  "
              f"DOWN={old.get('DOWN', 0):.1f}%  "
              f"SIDEWAYS={old.get('SIDEWAYS', 0):.1f}%")

    # Simple 24h forward return threshold — proven to work with this feature set
    df['target_direction'] = df['target_return'].apply(
        lambda x: 'UP' if x > thresh else ('DOWN' if x < -thresh else 'SIDEWAYS')
    )

    new = df['target_direction'].value_counts(normalize=True) * 100
    print(f"  New: UP={new.get('UP', 0):.1f}%  "
          f"DOWN={new.get('DOWN', 0):.1f}%  "
          f"SIDEWAYS={new.get('SIDEWAYS', 0):.1f}%")

    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved")


def main():
    coins = []
    for arg in sys.argv[1:]:
        coin = arg.upper()
        if '_USDT' not in coin:
            coin += '_USDT'
        if coin not in DIRECTION_THRESHOLDS:
            print(f"Unknown coin: {coin}. Valid: {ALL_COINS}")
            sys.exit(1)
        coins.append(coin)

    if not coins:
        coins = ALL_COINS

    print(f"Restoring simple threshold labels for: {coins}\n")
    for coin in coins:
        fix_coin(coin)

    print("\nDone. Re-run walk_forward_validation.py.")


if __name__ == '__main__':
    main()
