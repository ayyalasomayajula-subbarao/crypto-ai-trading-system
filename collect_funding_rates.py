"""
Fetch historical funding rates from Binance futures (fapi).
Endpoint: GET /fapi/v1/fundingRate
Period:   8h settlement (3× per day), paginated 1000 records per request.
Output:   data/{COIN}_funding.csv  (timestamp, funding_rate)

Run this BEFORE collect_multi_timeframe.py so the funding features are available.
Existing coins (BTC/ETH/SOL/PEPE) are skipped if their funding CSV already exists
unless you pass --force to overwrite.
"""

import os
import sys
import time
import requests
import pandas as pd

BASE_URL = 'https://fapi.binance.com/fapi/v1/fundingRate'

# Binance perpetual futures symbol + earliest start date to fetch from.
# AVAX/LINK perps launched Dec 2020; BNB perp launched Mar 2020.
# Existing coins kept here so you can refresh them with --force.
SYMBOLS = {
    'BTC_USDT':  ('BTCUSDT',  '2019-09-10'),
    'ETH_USDT':  ('ETHUSDT',  '2019-11-01'),
    'SOL_USDT':  ('SOLUSDT',  '2021-09-01'),
    'PEPE_USDT': ('1000PEPEUSDT', '2023-05-01'),
    'AVAX_USDT': ('AVAXUSDT', '2020-12-01'),
    'BNB_USDT':  ('BNBUSDT',  '2020-03-01'),
    'LINK_USDT': ('LINKUSDT', '2020-12-01'),
}

NEW_COINS = {'AVAX_USDT', 'BNB_USDT', 'LINK_USDT'}


def fetch_funding_history(futures_symbol: str, start_iso: str) -> list:
    since_ms = int(pd.Timestamp(start_iso).timestamp() * 1000)
    now_ms   = int(time.time() * 1000)
    rows     = []

    print(f"  Fetching funding rates for {futures_symbol} from {start_iso} ...")
    while since_ms < now_ms:
        try:
            r = requests.get(
                BASE_URL,
                params={
                    'symbol':    futures_symbol,
                    'startTime': since_ms,
                    'limit':     1000,
                },
                timeout=15,
            )
            data = r.json()
        except Exception as exc:
            print(f"    Request error: {exc} — retrying in 5s")
            time.sleep(5)
            continue

        if isinstance(data, dict):
            # Binance returns dict on error: {"code": -1121, "msg": "Invalid symbol."}
            print(f"    API error: {data.get('msg', data)}")
            break

        if not data:
            break

        rows.extend(data)
        last_ts = data[-1]['fundingTime']

        if len(data) < 1000:
            break  # reached end of available history

        since_ms = last_ts + 1
        time.sleep(0.3)

    print(f"    → {len(rows):,} records")
    return rows


def save_funding_csv(coin: str, rows: list) -> None:
    if not rows:
        print(f"  ⚠️  No data for {coin} — skipping")
        return

    df = pd.DataFrame(rows)
    df['timestamp']    = pd.to_datetime(df['fundingTime'], unit='ms')
    df['funding_rate'] = df['fundingRate'].astype(float)
    df = (
        df[['timestamp', 'funding_rate']]
        .sort_values('timestamp')
        .drop_duplicates('timestamp')
        .reset_index(drop=True)
    )

    os.makedirs('data', exist_ok=True)
    out_path = f'data/{coin}_funding.csv'
    df.to_csv(out_path, index=False)
    print(f"  ✅ Saved: {out_path}  ({len(df):,} rows, "
          f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()})")


def main():
    force = '--force' in sys.argv
    # If no args, only collect new coins. Pass --all to refresh all coins.
    collect_all = '--all' in sys.argv

    print("=" * 60)
    print("  Funding Rate History — Binance Futures")
    print("=" * 60)

    for coin, (futures_symbol, start_iso) in SYMBOLS.items():
        out_path = f'data/{coin}_funding.csv'
        is_new   = coin in NEW_COINS

        if not is_new and not collect_all and not force:
            if os.path.exists(out_path):
                print(f"\n[{coin}]  already exists — skipping (pass --all to refresh)")
                continue

        print(f"\n[{coin}]")
        rows = fetch_funding_history(futures_symbol, start_iso)
        save_funding_csv(coin, rows)

    print("\n✅ Done. Run collect_multi_timeframe.py next.")


if __name__ == '__main__':
    main()
