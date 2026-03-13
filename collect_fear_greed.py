"""
Fetch historical Crypto Fear & Greed Index from alternative.me.
Endpoint: https://api.alternative.me/fng/?limit=0
Returns:  daily scores 0-100 dating back to 2018-02-01.
Output:   data/fear_greed.csv  (single file — applies to all coins)

No API key required. limit=0 returns all available records.
"""

import os
import requests
import pandas as pd


def fetch_fear_greed() -> pd.DataFrame:
    print("Fetching Fear & Greed Index from alternative.me ...")
    r = requests.get(
        'https://api.alternative.me/fng/',
        params={'limit': 0, 'format': 'json'},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json().get('data', [])
    if not data:
        raise RuntimeError("Empty response from alternative.me")

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
    df['fng_value'] = df['value'].astype(int)
    df = (
        df[['timestamp', 'fng_value', 'value_classification']]
        .sort_values('timestamp')
        .drop_duplicates('timestamp')
        .reset_index(drop=True)
    )

    print(f"  → {len(df):,} records  ({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")
    return df


def main():
    df = fetch_fear_greed()

    os.makedirs('data', exist_ok=True)
    out = 'data/fear_greed.csv'
    df.to_csv(out, index=False)
    print(f"  ✅ Saved: {out}")


if __name__ == '__main__':
    main()
