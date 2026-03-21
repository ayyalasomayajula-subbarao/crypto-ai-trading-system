"""
NSE/BSE Data Collection — OHLCV via yfinance
Fetches 1h, 1d, 1w candles for all instruments.
4H is resampled from 1H (same lookahead-safe method as crypto).

Usage:
    python collect_nse_data.py                  # update all
    python collect_nse_data.py --symbol NIFTY50 # single instrument
    python collect_nse_data.py --full           # re-download full history
"""

from __future__ import annotations
import os
import sys
import argparse
import time
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import yfinance as yf
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INSTRUMENTS, ALL_SYMBOLS, DATA_DIR, ohlcv_path, TIMEZONE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

IST = pytz.timezone(TIMEZONE)

# ─── yfinance interval mappings ─────────────────────────────────────────────

YF_INTERVALS = {
    "1h": "1h",
    "1d": "1d",
    "1w": "1wk",
}

# yfinance 1h only goes back ~730 days; daily/weekly goes back further
MAX_LOOKBACK = {
    "1h":  729,    # days
    "1d":  5000,
    "1w":  5000,
}

# ─── Core fetch ─────────────────────────────────────────────────────────────

def fetch_ohlcv(yf_symbol: str, interval: str,
                start: str, end: str | None = None) -> pd.DataFrame:
    """Download OHLCV from yfinance, return clean DataFrame."""
    ticker = yf.Ticker(yf_symbol)
    yf_interval = YF_INTERVALS[interval]

    kwargs = {"interval": yf_interval, "start": start, "auto_adjust": True}
    if end:
        kwargs["end"] = end
    else:
        kwargs["end"] = datetime.now(IST).strftime("%Y-%m-%d")

    df = ticker.history(**kwargs)
    if df.empty:
        return pd.DataFrame()

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamp"

    # Ensure UTC-aware index, then convert to IST
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)

    # Drop rows with zero volume (market closed / holidays)
    df = df[df["volume"] > 0]
    df = df.sort_index()
    return df


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV to 4H candles (9:15 AM origin, IST-aware)."""
    if df_1h.empty:
        return pd.DataFrame()

    df = df_1h.copy()
    resampled = df.resample("4h", origin="start_day", offset="9h15min").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    # Keep only candles that started during market hours
    resampled = resampled[resampled.index.hour.isin([9, 13])]
    return resampled.sort_index()


# ─── Incremental update ─────────────────────────────────────────────────────

def _last_timestamp(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    try:
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
        if existing.empty:
            return None
        last = existing.index[-1]
        # Add 1 day buffer to avoid duplicates
        return (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        return None


def update_symbol(symbol: str, full: bool = False) -> dict:
    """Fetch and save all timeframes for one symbol."""
    cfg = INSTRUMENTS[symbol]
    yf_sym = cfg["yf_symbol"]
    data_start = cfg["data_start"]
    results = {}

    for tf in ["1h", "1d", "1w"]:
        path = ohlcv_path(symbol, tf)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if full:
            max_days = MAX_LOOKBACK[tf]
            start = (datetime.now() - timedelta(days=max_days)).strftime("%Y-%m-%d")
            start = max(start, data_start)
        else:
            start = _last_timestamp(path) or data_start
            # Cap 1H at 729 days back
            if tf == "1h":
                min_start = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
                start = max(start, min_start)

        try:
            df = fetch_ohlcv(yf_sym, tf, start)
            if df.empty:
                log.warning(f"{symbol} {tf}: no data returned")
                results[tf] = 0
                continue

            if not full and os.path.exists(path):
                existing = pd.read_csv(path, index_col=0, parse_dates=True)
                if existing.index.tz is None:
                    existing.index = existing.index.tz_localize(IST)
                combined = pd.concat([existing, df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                combined = df

            combined.to_csv(path)
            results[tf] = len(df)
            log.info(f"{symbol} {tf}: saved {len(df)} new rows → {len(combined)} total")
            time.sleep(0.3)  # rate limit

        except Exception as e:
            log.error(f"{symbol} {tf}: {e}")
            results[tf] = -1

    # Build 4H from 1H
    try:
        path_1h = ohlcv_path(symbol, "1h")
        path_4h = ohlcv_path(symbol, "4h")
        if os.path.exists(path_1h):
            df_1h = pd.read_csv(path_1h, index_col=0, parse_dates=True)
            if df_1h.index.tz is None:
                df_1h.index = df_1h.index.tz_localize(IST)
            df_4h = resample_to_4h(df_1h)
            df_4h.to_csv(path_4h)
            results["4h"] = len(df_4h)
            log.info(f"{symbol} 4h: resampled → {len(df_4h)} rows")
    except Exception as e:
        log.error(f"{symbol} 4h resample: {e}")
        results["4h"] = -1

    return {symbol: results}


# ─── Index-specific: advance/decline + market breadth ───────────────────────

def fetch_advance_decline() -> pd.DataFrame:
    """
    Approximate NSE advance/decline from NIFTY 500 constituent performance.
    Uses yfinance bulk download for a proxy set.
    """
    proxy_symbols = [
        "^NSEI", "^NSEBANK", "^CNXIT", "^CNXAUTO",
        "^CNXPHARMA", "^CNXENERGY", "^CNXFINANCE",
    ]
    dfs = []
    for sym in proxy_symbols:
        try:
            t = yf.Ticker(sym)
            h = t.history(period="1y", interval="1d", auto_adjust=True)
            if not h.empty:
                h = h[["Close"]].rename(columns={"Close": sym})
                dfs.append(h)
            time.sleep(0.2)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, axis=1)
    # Advance/decline ratio = sectors up / sectors total
    combined["advances"] = (combined.pct_change() > 0).sum(axis=1)
    combined["declines"] = (combined.pct_change() < 0).sum(axis=1)
    combined["ad_ratio"] = combined["advances"] / (combined["advances"] + combined["declines"] + 1e-9)
    result = combined[["advances", "declines", "ad_ratio"]].dropna()
    out_path = os.path.join(DATA_DIR, "advance_decline.csv")
    result.to_csv(out_path)
    log.info(f"Advance/Decline: saved {len(result)} rows")
    return result


# ─── GIFT Nifty overnight premium ────────────────────────────────────────────

def fetch_gift_nifty() -> pd.DataFrame:
    """
    GIFT Nifty (SGX Nifty successor) via yfinance proxy.
    Uses Dow futures as overnight global cue proxy when GIFT unavailable.
    """
    # NIFTY futures not available on yfinance; use US futures as proxy
    proxies = {
        "sp500_fut": "ES=F",   # S&P 500 futures
        "nasdaq_fut": "NQ=F",  # Nasdaq futures
        "dow_fut":   "YM=F",   # Dow futures
    }
    dfs = []
    for name, sym in proxies.items():
        try:
            t = yf.Ticker(sym)
            h = t.history(period="2y", interval="1d", auto_adjust=True)
            if not h.empty:
                h = h[["Close"]].rename(columns={"Close": name})
                dfs.append(h)
            time.sleep(0.2)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, axis=1)
    # overnight_return = average return of US futures (proxy for GIFT signal)
    combined["gift_proxy_return"] = combined.pct_change().mean(axis=1)
    out_path = os.path.join(DATA_DIR, "gift_nifty_proxy.csv")
    combined.to_csv(out_path)
    log.info(f"GIFT Nifty proxy: saved {len(combined)} rows")
    return combined


# ─── India VIX ───────────────────────────────────────────────────────────────

def fetch_india_vix() -> pd.DataFrame:
    """Fetch India VIX from yfinance."""
    from config import INDIA_VIX_PATH
    try:
        t = yf.Ticker("^INDIAVIX")
        df = t.history(period="5y", interval="1d", auto_adjust=True)
        if df.empty:
            # fallback symbol
            t2 = yf.Ticker("INDIAVIX.NS")
            df = t2.history(period="5y", interval="1d", auto_adjust=True)

        if df.empty:
            log.warning("India VIX: no data from yfinance")
            return pd.DataFrame()

        df = df[["Close"]].rename(columns={"Close": "india_vix"})
        df.index.name = "timestamp"
        df.to_csv(INDIA_VIX_PATH)
        log.info(f"India VIX: saved {len(df)} rows")
        return df
    except Exception as e:
        log.error(f"India VIX fetch failed: {e}")
        return pd.DataFrame()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NSE/BSE data collector")
    parser.add_argument("--symbol",  type=str,  default=None,
                        help="Single symbol to update (e.g. NIFTY50)")
    parser.add_argument("--full",    action="store_true",
                        help="Full re-download (ignore existing data)")
    parser.add_argument("--workers", type=int,  default=4,
                        help="Parallel workers")
    parser.add_argument("--vix",     action="store_true",
                        help="Also fetch India VIX")
    parser.add_argument("--ad",      action="store_true",
                        help="Also fetch advance/decline data")
    parser.add_argument("--gift",    action="store_true",
                        help="Also fetch GIFT Nifty proxy")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    _active_env = os.environ.get("ACTIVE_SYMBOLS", "")
    _active = [s.strip() for s in _active_env.split(",") if s.strip()] if _active_env else ALL_SYMBOLS
    symbols = [args.symbol] if args.symbol else _active

    log.info(f"Updating {len(symbols)} symbols  full={args.full}  workers={args.workers}")

    # Parallel OHLCV collection
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(update_symbol, s, args.full): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                result = fut.result()
                log.info(f"Done: {sym} → {result[sym]}")
            except Exception as e:
                log.error(f"Failed: {sym} → {e}")

    if args.vix or not args.symbol:
        fetch_india_vix()

    if args.ad or not args.symbol:
        fetch_advance_decline()

    if args.gift or not args.symbol:
        fetch_gift_nifty()

    log.info("Data collection complete.")


if __name__ == "__main__":
    main()
