"""
Fetch historical Open Interest (OI) from Binance futures.
Endpoint: https://fapi.binance.com/futures/data/openInterestHist
Period:   1h bars, paginated 500 records per request.
Output:   data/{COIN}_USDT_openinterest.csv  (timestamp, open_interest)

Note: open_interest is returned in "sumOpenInterestValue" (USD notional).
PEPE futures on Binance use the 1000PEPEUSDT symbol (1000x multiplier prefix).
"""

import os
import time
import requests
import pandas as pd

BASE_URL = 'https://fapi.binance.com/futures/data/openInterestHist'

# Binance futures symbol → (futures_ticker, earliest_start_date)
SYMBOLS = {
    'BTC_USDT':  ('BTCUSDT',      '2019-09-01'),
    'ETH_USDT':  ('ETHUSDT',      '2019-11-01'),
    'SOL_USDT':  ('SOLUSDT',      '2021-09-01'),
    'PEPE_USDT': ('1000PEPEUSDT', '2023-05-01'),
}


def fetch_oi_history(futures_symbol: str, start_iso: str) -> list:
    since_ms = int(pd.Timestamp(start_iso).timestamp() * 1000)
    now_ms   = int(time.time() * 1000)
    rows     = []

    print(f"  Fetching OI for {futures_symbol} from {start_iso} ...")
    while since_ms < now_ms:
        try:
            r = requests.get(
                BASE_URL,
                params={
                    'symbol':    futures_symbol,
                    'period':    '1h',
                    'limit':     500,
                    'startTime': since_ms,
                },
                timeout=10,
            )
            data = r.json()
        except Exception as exc:
            print(f"    Request error: {exc} — retrying in 5s")
            time.sleep(5)
            continue

        if isinstance(data, dict):
            # Binance returns a dict on error, e.g. {"code": -1121, "msg": "Invalid symbol."}
            print(f"    API error response: {data}")
            break

        if not data:
            break

        rows.extend(data)
        last_ts = data[-1]['timestamp']

        if len(data) < 500:
            # Reached the end of available history
            break

        since_ms = last_ts + 3_600_000  # advance one hour
        time.sleep(0.4)                  # stay within Binance rate limits

    print(f"    → {len(rows):,} records")
    return rows


def save_oi_csv(coin: str, rows: list) -> None:
    if not rows:
        print(f"  ⚠️  No data for {coin} — skipping save")
        return

    df = pd.DataFrame(rows)
    # Binance field names: timestamp (ms epoch), sumOpenInterest (contracts),
    # sumOpenInterestValue (USD notional).  We want USD notional.
    df['timestamp']     = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df['open_interest'] = df['sumOpenInterestValue'].astype(float)
    df = df[['timestamp', 'open_interest']].sort_values('timestamp').drop_duplicates('timestamp')

    os.makedirs('data', exist_ok=True)
    out_path = f'data/{coin}_openinterest.csv'
    df.to_csv(out_path, index=False)
    print(f"  ✅ Saved: {out_path}  ({len(df):,} rows, "
          f"{df['timestamp'].min()} → {df['timestamp'].max()})")

    # Data-depth warning: if history is shallower than 1 year, OI features
    # will cause dropna() to remove years of training rows.
    earliest = df['timestamp'].min()
    cutoff   = pd.Timestamp('today') - pd.DateOffset(years=1)
    if earliest > cutoff:
        months = (pd.Timestamp('today') - earliest).days // 30
        print(f"  ⚠️  WARNING: only {months} months of OI history for {coin}. "
              "Consider skipping OI features for this coin.")


def main():
    print("=" * 60)
    print("  Open Interest History — Binance Futures")
    print("=" * 60)

    for coin, (futures_symbol, start_iso) in SYMBOLS.items():
        print(f"\n[{coin}]")
        rows = fetch_oi_history(futures_symbol, start_iso)
        save_oi_csv(coin, rows)

    print("\nDone.")


if __name__ == '__main__':
    main()
