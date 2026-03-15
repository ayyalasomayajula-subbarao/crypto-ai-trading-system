"""
collect_taker_volume.py — fetch taker buy/sell volume from Binance REST API.

Binance klines col 9 = taker_buy_base_asset_volume (what aggressive buyers bought).
ccxt strips this to 5 columns; we use the raw REST endpoint to get all 12 kline cols.

Output: data/{COIN}_taker_volume.csv
Columns: timestamp, volume, taker_buy_volume, taker_sell_volume

Run:
    python collect_taker_volume.py            # all 10 coins, incremental
    python collect_taker_volume.py BTC_USDT   # single coin
    python collect_taker_volume.py --full     # full history refetch
"""

import requests
import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timezone

COINS = [
    'BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT',
    'AVAX_USDT', 'BNB_USDT', 'LINK_USDT',
    'ARB_USDT', 'OP_USDT', 'INJ_USDT',
]

BASE_URL = 'https://api.binance.com/api/v3/klines'
INTERVAL = '1h'
LIMIT    = 1000        # Binance max per request
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Binance kline column positions (0-indexed)
# 0:open_time 1:open 2:high 3:low 4:close 5:volume 6:close_time
# 7:quote_asset_volume 8:num_trades 9:taker_buy_base 10:taker_buy_quote 11:ignore
COL_OPEN_TIME        = 0
COL_VOLUME           = 5
COL_TAKER_BUY_BASE   = 9


def _coin_to_symbol(coin: str) -> str:
    return coin.replace('_', '')   # BTC_USDT → BTCUSDT


def _fetch_batch(symbol: str, start_ms: int) -> list:
    """Fetch up to LIMIT klines starting from start_ms (Unix ms)."""
    params = {
        'symbol':    symbol,
        'interval':  INTERVAL,
        'startTime': start_ms,
        'limit':     LIMIT,
    }
    resp = requests.get(BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_taker_volume(coin: str, since_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Fetch 1h klines for coin from Binance REST.

    Parameters
    ----------
    coin      : e.g. 'BTC_USDT'
    since_ts  : fetch from this timestamp onward (for incremental updates).
                None = fetch full history.

    Returns
    -------
    DataFrame with columns: timestamp, volume, taker_buy_volume, taker_sell_volume
    All floats, timestamp is UTC naive.
    """
    symbol = _coin_to_symbol(coin)

    if since_ts is not None:
        start_ms = int(since_ts.timestamp() * 1000)
    else:
        # Binance launches vary; start from a safe epoch
        start_ms = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    rows = []
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f"  Fetching {coin} from {pd.Timestamp(start_ms, unit='ms')} …", flush=True)

    while start_ms < now_ms:
        batch = _fetch_batch(symbol, start_ms)
        if not batch:
            break

        for k in batch:
            open_time_ms    = int(k[COL_OPEN_TIME])
            total_vol       = float(k[COL_VOLUME])
            taker_buy_vol   = float(k[COL_TAKER_BUY_BASE])
            taker_sell_vol  = total_vol - taker_buy_vol
            rows.append((open_time_ms, total_vol, taker_buy_vol, taker_sell_vol))

        last_ts = batch[-1][COL_OPEN_TIME]
        start_ms = int(last_ts) + 3_600_000   # advance by 1h

        if len(batch) < LIMIT:
            break   # reached the end

        time.sleep(0.12)   # ~8 req/s, well within Binance 1200/min limit

    if not rows:
        return pd.DataFrame(columns=['timestamp', 'volume', 'taker_buy_volume', 'taker_sell_volume'])

    df = pd.DataFrame(rows, columns=['timestamp_ms', 'volume', 'taker_buy_volume', 'taker_sell_volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.drop(columns=['timestamp_ms'])
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def update_coin(coin: str, full: bool = False) -> int:
    """
    Incremental update: load existing CSV, fetch only new rows, append and save.

    Returns number of new rows added.
    """
    out_path = os.path.join(DATA_DIR, f'{coin}_taker_volume.csv')

    since_ts = None
    existing = pd.DataFrame()

    if not full and os.path.exists(out_path):
        existing = pd.read_csv(out_path, parse_dates=['timestamp'])
        if not existing.empty:
            last_ts   = existing['timestamp'].max()
            since_ts  = last_ts   # re-fetch from last known row (dedup handles overlap)
            print(f"  Existing: {len(existing)} rows up to {last_ts}")

    new_df = fetch_taker_volume(coin, since_ts=since_ts)

    if new_df.empty:
        print(f"  No new data for {coin}")
        return 0

    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    else:
        combined = new_df

    combined.to_csv(out_path, index=False)
    new_rows = len(combined) - len(existing)
    print(f"  Saved {len(combined)} rows (+{max(new_rows, 0)} new) → {out_path}")
    return max(new_rows, 0)


def main():
    full = '--full' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    coins = args if args else COINS

    os.makedirs(DATA_DIR, exist_ok=True)
    total_new = 0

    for coin in coins:
        print(f"\n{'='*50}")
        print(f"  {coin}")
        print(f"{'='*50}")
        try:
            n = update_coin(coin, full=full)
            total_new += n
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. Total new rows: {total_new}")


if __name__ == '__main__':
    main()
