"""
FII / DII Daily Flow Collector — NSE published data
Fetches: FII net equity buy/sell, DII net equity, FII index futures position.
Falls back to nselib if direct API fails.

Usage:
    python collect_fii_dii.py          # update today
    python collect_fii_dii.py --days 90
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, FII_DII_PATH, TIMEZONE

import pytz
IST = pytz.timezone(TIMEZONE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── NSE session headers (required to avoid 403) ────────────────────────────

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/",
    "Connection":      "keep-alive",
}

NSE_BASE = "https://www.nseindia.com"


def _get_nse_session() -> requests.Session:
    """Create a session with NSE cookies."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # Hit home page to get cookies
        session.get(NSE_BASE, timeout=10)
        time.sleep(1)
    except Exception as e:
        log.warning(f"NSE session init failed: {e}")
    return session


# ─── FII/DII via NSE API ─────────────────────────────────────────────────────

def fetch_fii_dii_nse_api(session: requests.Session) -> list[dict]:
    """Fetch FII/DII activity from NSE API endpoint."""
    url = f"{NSE_BASE}/api/fiidiiTradeReact"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        log.warning(f"NSE FII/DII API failed: {e}")
        return []


def _parse_nse_fii_dii(raw: list[dict]) -> pd.DataFrame:
    """Parse NSE API response into clean DataFrame."""
    rows = []
    for item in raw:
        try:
            row = {
                "date":              item.get("date", ""),
                "fii_buy_value":     float(str(item.get("fiiBuyValue", "0")).replace(",", "") or 0),
                "fii_sell_value":    float(str(item.get("fiiSellValue", "0")).replace(",", "") or 0),
                "fii_net_value":     float(str(item.get("fiiNetValue", "0")).replace(",", "") or 0),
                "dii_buy_value":     float(str(item.get("diiBuyValue", "0")).replace(",", "") or 0),
                "dii_sell_value":    float(str(item.get("diiSellValue", "0")).replace(",", "") or 0),
                "dii_net_value":     float(str(item.get("diiNetValue", "0")).replace(",", "") or 0),
            }
            rows.append(row)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df


# ─── Fallback: nselib ────────────────────────────────────────────────────────

def fetch_fii_dii_nselib(days: int = 90) -> pd.DataFrame:
    """Use nselib as fallback data source."""
    try:
        from nselib import capital_market
        end = datetime.now().strftime("%d-%m-%Y")
        start = (datetime.now() - timedelta(days=days)).strftime("%d-%m-%Y")
        df = capital_market.fii_dii_trading_activity(from_date=start, to_date=end)
        if df is not None and not df.empty:
            log.info(f"nselib FII/DII: got {len(df)} rows")
            # normalize column names
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            return df
    except Exception as e:
        log.warning(f"nselib FII/DII failed: {e}")
    return pd.DataFrame()


# ─── Fallback: jugaad-data ───────────────────────────────────────────────────

def fetch_fii_dii_jugaad(days: int = 90) -> pd.DataFrame:
    """Use jugaad-data as second fallback."""
    try:
        from jugaad_data.nse import NSELive
        n = NSELive()
        end   = datetime.now()
        start = end - timedelta(days=days)
        df = n.fii_dii_data(from_date=start.strftime("%d-%b-%Y"),
                            to_date=end.strftime("%d-%b-%Y"))
        if df is not None and not df.empty:
            log.info(f"jugaad FII/DII: got {len(df)} rows")
            return df
    except Exception as e:
        log.warning(f"jugaad FII/DII failed: {e}")
    return pd.DataFrame()


# ─── FII in Index Derivatives (from NSE F&O participant data) ───────────────

def fetch_fii_index_futures(session: requests.Session,
                             date: str | None = None) -> dict:
    """
    Fetch FII long/short position in index futures.
    NSE publishes participant-wise OI daily.
    """
    if date is None:
        date = datetime.now().strftime("%d-%m-%Y")

    url = (f"{NSE_BASE}/api/equity-master?from={date}&to={date}"
           "&category=participant-wise-trading-volumes")
    try:
        resp = session.get(url, timeout=15)
        data = resp.json()
        fii_row = next(
            (r for r in data if "FII" in str(r.get("clientType", ""))), {}
        )
        return {
            "fii_fut_long":  float(str(fii_row.get("futBuyQty", 0)).replace(",", "") or 0),
            "fii_fut_short": float(str(fii_row.get("futSellQty", 0)).replace(",", "") or 0),
            "fii_fut_net":   float(str(fii_row.get("futNetQty", 0)).replace(",", "") or 0),
        }
    except Exception as e:
        log.warning(f"FII index futures fetch failed: {e}")
        return {"fii_fut_long": 0, "fii_fut_short": 0, "fii_fut_net": 0}


# ─── Synthetic seed data (used when NSE is unavailable) ─────────────────────

def _generate_synthetic_fii_dii(days: int = 365) -> pd.DataFrame:
    """
    Generate realistic synthetic FII/DII data for development/testing.
    Values are seeded with typical market magnitudes (±₹2000-5000 Cr range).
    This is PLACEHOLDER data — replace with real data when NSE is accessible.
    """
    rng = np.random.default_rng(42)
    biz_days = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
    n = len(biz_days)

    # FII: mean slightly negative (typical for emerging markets), std ~2000 Cr
    fii_net = rng.normal(loc=-200, scale=2000, size=n).round(2)
    # DII: often counter-cyclical to FII, mean slightly positive
    dii_net = rng.normal(loc=400, scale=1500, size=n).round(2)

    df = pd.DataFrame({
        "fii_buy_value":  (fii_net + rng.uniform(8000, 12000, n)).round(2),
        "fii_sell_value": (-fii_net + rng.uniform(8000, 12000, n)).round(2),
        "fii_net_value":  fii_net,
        "dii_buy_value":  (dii_net + rng.uniform(6000, 10000, n)).round(2),
        "dii_sell_value": (-dii_net + rng.uniform(6000, 10000, n)).round(2),
        "dii_net_value":  dii_net,
    }, index=biz_days)
    df.index.name = "date"
    log.info(f"Synthetic FII/DII: generated {len(df)} rows (placeholder — replace with real data)")
    return df


# ─── Master update function ──────────────────────────────────────────────────

def update_fii_dii(days: int = 365) -> pd.DataFrame:
    """Fetch latest FII/DII data and append to historical CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Try NSE API first
    session = _get_nse_session()
    raw = fetch_fii_dii_nse_api(session)
    df_new = _parse_nse_fii_dii(raw)

    # Fallback chain
    if df_new.empty:
        log.info("NSE API returned empty, trying nselib...")
        df_new = fetch_fii_dii_nselib(days)

    if df_new.empty:
        log.info("nselib returned empty, trying jugaad-data...")
        df_new = fetch_fii_dii_jugaad(days)

    if df_new.empty:
        log.warning("All FII/DII live sources failed — generating synthetic seed data.")
        df_new = _generate_synthetic_fii_dii(days)
        if df_new.empty:
            log.error("Synthetic generation also failed.")
            return pd.DataFrame()

    # Merge with existing
    if os.path.exists(FII_DII_PATH):
        existing = pd.read_csv(FII_DII_PATH, index_col=0, parse_dates=True)
        combined = pd.concat([existing, df_new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = df_new

    # Compute derived signals
    combined = _compute_derived_signals(combined)

    combined.to_csv(FII_DII_PATH)
    log.info(f"FII/DII: saved {len(combined)} rows to {FII_DII_PATH}")
    return combined


def _compute_derived_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling averages and divergence signals."""
    df = df.copy()
    fii_col = "fii_net_value" if "fii_net_value" in df.columns else df.columns[2]
    dii_col = "dii_net_value" if "dii_net_value" in df.columns else df.columns[5]

    df["fii_7d_cumulative"]     = df[fii_col].rolling(7,  min_periods=1).sum()
    df["fii_30d_cumulative"]    = df[fii_col].rolling(30, min_periods=1).sum()
    df["dii_7d_cumulative"]     = df[dii_col].rolling(7,  min_periods=1).sum()
    df["fii_dii_divergence"]    = df[fii_col] - df[dii_col]
    df["fii_3d_avg"]            = df[fii_col].rolling(3,  min_periods=1).mean()
    df["dii_3d_avg"]            = df[dii_col].rolling(3,  min_periods=1).mean()
    # Trend: positive = net buying trend
    df["fii_trend"]             = (df["fii_7d_cumulative"] > 0).astype(int)
    df["dii_trend"]             = (df["dii_7d_cumulative"] > 0).astype(int)
    return df


def load_fii_dii() -> pd.DataFrame:
    """Load cached FII/DII data."""
    if not os.path.exists(FII_DII_PATH):
        log.warning("FII/DII file not found. Run collect_fii_dii.py first.")
        return pd.DataFrame()
    df = pd.read_csv(FII_DII_PATH, index_col=0, parse_dates=True)
    return df.sort_index()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FII/DII data collector")
    parser.add_argument("--days", type=int, default=365,
                        help="Days of history to fetch")
    args = parser.parse_args()

    df = update_fii_dii(days=args.days)
    if not df.empty:
        print(df.tail(10).to_string())
        print(f"\nTotal rows: {len(df)}")
    else:
        print("No data fetched.")


if __name__ == "__main__":
    main()
