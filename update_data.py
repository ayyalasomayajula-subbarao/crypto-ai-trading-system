#!/usr/bin/env python3
"""
Incremental Data Updater for Crypto AI Trading System

Instead of re-downloading everything from 2017, this script:
1. Reads the last timestamp from each CSV
2. Fetches only NEW candles from Binance
3. Appends them to the existing CSVs
4. Regenerates multi-timeframe features

Usage:
    python update_data.py              # Update all coins
    python update_data.py --coin BTC   # Update specific coin
    python update_data.py --full       # Full re-download (like collect_multi_timeframe.py)

Automation (cron):
    # Run every 6 hours
    crontab -e
    0 */6 * * * cd /Users/subbuayyalasomayajula/Desktop/crypto-ai-system && /Users/subbuayyalasomayajula/Desktop/crypto-ai-system/venv/bin/python update_data.py >> data/update.log 2>&1
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys
import argparse

# Configuration
COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'PEPE/USDT']
TIMEFRAMES = ['1h', '4h', '1d', '1w']
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class IncrementalUpdater:
    def __init__(self):
        self.exchange = ccxt.binance()
        os.makedirs(DATA_DIR, exist_ok=True)

    def get_last_timestamp(self, coin: str, timeframe: str) -> int:
        """Read the last timestamp from an existing CSV file.
        Returns millisecond timestamp, or 0 if file doesn't exist."""
        filename = os.path.join(DATA_DIR, f"{coin}_{timeframe}.csv")
        if not os.path.exists(filename):
            return 0

        try:
            df = pd.read_csv(filename)
            if len(df) == 0:
                return 0
            last_ts = pd.to_datetime(df['timestamp'].iloc[-1])
            return int(last_ts.timestamp() * 1000)
        except Exception as e:
            print(f"    Warning: Could not read {filename}: {e}")
            return 0

    def fetch_new_candles(self, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
        """Fetch only candles newer than since_ms from Binance."""
        all_ohlcv = []
        since = since_ms + 1  # Start after last known candle

        while since < self.exchange.milliseconds():
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1

                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                print(f"    Error fetching {symbol} {timeframe}: {e}")
                time.sleep(5)
                continue

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        return df

    def update_coin_timeframe(self, symbol: str, timeframe: str, full: bool = False) -> int:
        """Update a single coin/timeframe. Returns number of new candles added."""
        coin = symbol.replace('/', '_')
        filename = os.path.join(DATA_DIR, f"{coin}_{timeframe}.csv")

        if full or not os.path.exists(filename):
            # Full download from 2017
            print(f"  [{timeframe}] Full download...")
            since = self.exchange.parse8601('2017-01-01T00:00:00Z')
            new_df = self.fetch_new_candles(symbol, timeframe, since - 1)
            if len(new_df) > 0:
                new_df.to_csv(filename, index=False)
                print(f"    Downloaded {len(new_df):,} candles")
                return len(new_df)
            return 0

        # Incremental update
        last_ts = self.get_last_timestamp(coin, timeframe)
        if last_ts == 0:
            return self.update_coin_timeframe(symbol, timeframe, full=True)

        last_dt = datetime.utcfromtimestamp(last_ts / 1000)
        hours_behind = (datetime.utcnow() - last_dt).total_seconds() / 3600
        print(f"  [{timeframe}] Last data: {last_dt.strftime('%Y-%m-%d %H:%M')} ({hours_behind:.0f}h behind)")

        new_df = self.fetch_new_candles(symbol, timeframe, last_ts)

        if len(new_df) == 0:
            print(f"    Already up to date")
            return 0

        # Append to existing CSV
        existing_df = pd.read_csv(filename)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        combined.to_csv(filename, index=False)

        print(f"    +{len(new_df)} new candles (total: {len(combined):,})")
        return len(new_df)

    def update_coin(self, symbol: str, full: bool = False) -> dict:
        """Update all timeframes for a coin."""
        coin = symbol.replace('/', '_')
        print(f"\n{'='*50}")
        print(f"  Updating {symbol}")
        print(f"{'='*50}")

        results = {}
        for tf in TIMEFRAMES:
            count = self.update_coin_timeframe(symbol, tf, full)
            results[tf] = count

        return results

    def regenerate_features(self, coin: str):
        """Regenerate multi-timeframe features after data update."""
        # Import feature generation from existing script
        try:
            from collect_multi_timeframe import create_multi_timeframe_features
            print(f"\n  Regenerating features for {coin}...")
            create_multi_timeframe_features(coin)
        except Exception as e:
            print(f"  Warning: Could not regenerate features for {coin}: {e}")
            print(f"  You can manually run: python collect_multi_timeframe.py")

    def update_all(self, full: bool = False, coins: list = None):
        """Update all coins and regenerate features."""
        target_coins = coins or COINS
        start_time = time.time()

        print("\n" + "#" * 60)
        print(f"  DATA UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Mode: {'FULL DOWNLOAD' if full else 'INCREMENTAL'}")
        print(f"  Coins: {', '.join(target_coins)}")
        print("#" * 60)

        total_new = 0
        updated_coins = []

        for symbol in target_coins:
            results = self.update_coin(symbol, full)
            coin_new = sum(results.values())
            total_new += coin_new
            if coin_new > 0:
                updated_coins.append(symbol.replace('/', '_'))

        # Regenerate features for coins that got new data
        if updated_coins:
            print("\n" + "=" * 60)
            print("  REGENERATING FEATURES")
            print("=" * 60)
            for coin in updated_coins:
                self.regenerate_features(coin)

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"  UPDATE COMPLETE")
        print(f"  New candles: {total_new:,}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Coins updated: {', '.join(updated_coins) if updated_coins else 'None (all up to date)'}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Update crypto data incrementally')
    parser.add_argument('--full', action='store_true', help='Full re-download from 2017')
    parser.add_argument('--coin', type=str, help='Update specific coin (e.g., BTC, ETH, SOL, PEPE)')
    parser.add_argument('--no-features', action='store_true', help='Skip feature regeneration')
    args = parser.parse_args()

    updater = IncrementalUpdater()

    coins = None
    if args.coin:
        coin_upper = args.coin.upper()
        coin_symbol = f"{coin_upper}/USDT"
        if coin_symbol in COINS:
            coins = [coin_symbol]
        else:
            print(f"Unknown coin: {args.coin}. Available: {', '.join(c.split('/')[0] for c in COINS)}")
            sys.exit(1)

    updater.update_all(full=args.full, coins=coins)


if __name__ == "__main__":
    main()
