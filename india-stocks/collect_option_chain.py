"""
NSE Option Chain Collector — PCR, Open Interest, Max Pain
Fetches live option chain and derives key F&O signals.

Usage:
    python collect_option_chain.py                # all index instruments
    python collect_option_chain.py --symbol NIFTY
    python collect_option_chain.py --history      # save daily snapshots
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, date

import pandas as pd
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, OPTION_CHAIN_DIR, TIMEZONE

import pytz
IST = pytz.timezone(TIMEZONE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NSE_BASE = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":           "application/json, text/plain, */*",
    "Accept-Language":  "en-US,en;q=0.9",
    "Accept-Encoding":  "gzip, deflate, br",
    "Referer":          "https://www.nseindia.com/option-chain",
    "Connection":       "keep-alive",
    "sec-ch-ua":        '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest":   "empty",
    "sec-fetch-mode":   "cors",
    "sec-fetch-site":   "same-origin",
}

# Map our symbol names to NSE option chain symbol names
NSE_OC_SYMBOLS = {
    "NIFTY50":   "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "NIFTYIT":   "NIFTYIT",
}


# ─── Session management ──────────────────────────────────────────────────────

def _new_session() -> requests.Session:
    """Two-step session init: homepage → option-chain page → API works."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # Step 1: hit homepage to get initial cookies
        session.get(f"{NSE_BASE}/", timeout=15)
        time.sleep(2)
        # Step 2: hit option-chain page to get dynamic cookies
        session.get(f"{NSE_BASE}/option-chain", timeout=15)
        time.sleep(2)
    except Exception as e:
        log.warning(f"Session init: {e}")
    return session


# ─── Raw option chain fetch ──────────────────────────────────────────────────

def fetch_raw_option_chain(nse_symbol: str,
                            session: requests.Session | None = None) -> dict:
    """
    Fetch option chain JSON from NSE.
    Returns raw data dict or empty dict on failure.
    """
    if session is None:
        session = _new_session()

    url = f"{NSE_BASE}/api/option-chain-indices?symbol={nse_symbol}"
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if "403" in str(e):
            log.warning(f"NSE 403 for {nse_symbol} — refreshing session")
            session = _new_session()
            try:
                resp = session.get(url, timeout=20)
                return resp.json()
            except Exception:
                pass
        log.error(f"Option chain HTTP error {nse_symbol}: {e}")
        return {}
    except Exception as e:
        log.error(f"Option chain fetch failed {nse_symbol}: {e}")
        return {}


# ─── Parse option chain ──────────────────────────────────────────────────────

def parse_option_chain(raw: dict) -> pd.DataFrame:
    """
    Parse NSE option chain into a strike-level DataFrame.
    Columns: strike, ce_oi, pe_oi, ce_iv, pe_iv, ce_ltp, pe_ltp,
             ce_change_oi, pe_change_oi
    """
    try:
        records = raw.get("records", {}).get("data", [])
    except Exception:
        return pd.DataFrame()

    rows = []
    for item in records:
        strike = item.get("strikePrice", 0)
        ce = item.get("CE", {}) or {}
        pe = item.get("PE", {}) or {}
        rows.append({
            "strike":        strike,
            "ce_oi":         ce.get("openInterest", 0) or 0,
            "pe_oi":         pe.get("openInterest", 0) or 0,
            "ce_iv":         ce.get("impliedVolatility", 0) or 0,
            "pe_iv":         pe.get("impliedVolatility", 0) or 0,
            "ce_ltp":        ce.get("lastPrice", 0) or 0,
            "pe_ltp":        pe.get("lastPrice", 0) or 0,
            "ce_change_oi":  ce.get("changeinOpenInterest", 0) or 0,
            "pe_change_oi":  pe.get("changeinOpenInterest", 0) or 0,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    return df


# ─── Derived metrics ─────────────────────────────────────────────────────────

def compute_pcr(df: pd.DataFrame) -> float:
    """Put/Call Ratio by open interest."""
    total_ce = df["ce_oi"].sum()
    total_pe = df["pe_oi"].sum()
    if total_ce == 0:
        return 1.0
    return round(total_pe / total_ce, 4)


def compute_max_pain(df: pd.DataFrame) -> float:
    """
    Max Pain: strike where total options sellers' loss is minimised.
    = strike that minimises sum of (intrinsic value × OI) for all options.
    """
    if df.empty:
        return 0.0

    strikes = df["strike"].values
    ce_oi   = df["ce_oi"].values
    pe_oi   = df["pe_oi"].values
    total_pain = []

    for s in strikes:
        ce_pain = np.sum(np.maximum(s - strikes, 0) * ce_oi)
        pe_pain = np.sum(np.maximum(strikes - s, 0) * pe_oi)
        total_pain.append(ce_pain + pe_pain)

    return float(strikes[np.argmin(total_pain)])


def compute_iv_percentile(iv_series: pd.Series, window: int = 252) -> float:
    """IV percentile rank vs last 252 trading days (0-100)."""
    if len(iv_series) < 10:
        return 50.0
    recent = iv_series.dropna()
    if len(recent) == 0:
        return 50.0
    current = float(recent.iloc[-1])
    lookback = recent.tail(window)
    rank = (lookback < current).sum() / len(lookback) * 100
    return round(rank, 1)


def get_underlying_price(raw: dict) -> float:
    """Extract underlying spot price from option chain data."""
    try:
        return float(raw.get("records", {}).get("underlyingValue", 0))
    except Exception:
        return 0.0


# ─── Historical PCR snapshot ─────────────────────────────────────────────────

def _pcr_snapshot_path(symbol: str) -> str:
    os.makedirs(OPTION_CHAIN_DIR, exist_ok=True)
    return os.path.join(OPTION_CHAIN_DIR, f"{symbol}_pcr_history.csv")


def update_pcr_history(symbol: str, pcr: float, underlying: float,
                        max_pain: float) -> pd.DataFrame:
    """Append today's PCR snapshot to history CSV."""
    path = _pcr_snapshot_path(symbol)
    today = pd.Timestamp(datetime.now(IST).date())

    new_row = pd.DataFrame([{
        "date":       today,
        "pcr":        pcr,
        "underlying": underlying,
        "max_pain":   max_pain,
        "max_pain_dist_pct": round(
            (underlying - max_pain) / max_pain * 100 if max_pain else 0, 3
        ),
    }]).set_index("date")

    if os.path.exists(path):
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
        combined = pd.concat([existing, new_row])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_row

    combined.to_csv(path)
    return combined


def load_pcr_history(symbol: str) -> pd.DataFrame:
    path = _pcr_snapshot_path(symbol)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ─── Full IV history for IV percentile ───────────────────────────────────────

def _iv_history_path(symbol: str) -> str:
    os.makedirs(OPTION_CHAIN_DIR, exist_ok=True)
    return os.path.join(OPTION_CHAIN_DIR, f"{symbol}_iv_history.csv")


def update_iv_history(symbol: str, atm_ce_iv: float, atm_pe_iv: float) -> None:
    path = _iv_history_path(symbol)
    today = pd.Timestamp(datetime.now(IST).date())
    new_row = pd.DataFrame([{
        "date":       today,
        "atm_ce_iv":  atm_ce_iv,
        "atm_pe_iv":  atm_pe_iv,
        "avg_iv":     (atm_ce_iv + atm_pe_iv) / 2,
    }]).set_index("date")

    if os.path.exists(path):
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
        combined = pd.concat([existing, new_row])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_row

    combined.to_csv(path)


# ─── Master update ───────────────────────────────────────────────────────────

_SYNTHETIC_UNDERLYING = {"NIFTY50": 22500.0, "BANKNIFTY": 48000.0, "NIFTYIT": 34000.0}

def _synthetic_oc_metrics(symbol: str) -> dict:
    """Return placeholder option chain metrics when NSE is unavailable."""
    underlying = _SYNTHETIC_UNDERLYING.get(symbol, 22500.0)
    log.warning(f"{symbol}: using synthetic option chain (NSE blocked) — run during market hours")
    return {
        "symbol":           symbol,
        "timestamp":        datetime.now(IST).isoformat(),
        "underlying":       underlying,
        "pcr":              1.05,
        "pcr_3d_avg":       1.05,
        "pcr_7d_avg":       1.05,
        "max_pain":         round(underlying * 0.995, 0),
        "max_pain_dist_pct": 0.5,
        "atm_ce_iv":        15.0,
        "atm_pe_iv":        16.0,
        "iv_percentile":    50.0,
        "total_ce_oi":      0,
        "total_pe_oi":      0,
        "ce_oi_change":     0.0,
        "pe_oi_change":     0.0,
        "oi_change_pct":    0.0,
        "_synthetic":       True,
    }


def update_symbol_oc(symbol: str, save_history: bool = True) -> dict:
    """
    Fetch option chain for one symbol, compute all metrics, save history.
    Returns dict of computed metrics.
    """
    nse_sym = NSE_OC_SYMBOLS.get(symbol, symbol.replace("NIFTY50", "NIFTY"))
    session = _new_session()

    raw = fetch_raw_option_chain(nse_sym, session)
    if not raw:
        return _synthetic_oc_metrics(symbol)

    df = parse_option_chain(raw)
    if df.empty:
        log.error(f"Empty parsed option chain for {symbol}")
        return {}

    underlying  = get_underlying_price(raw)
    pcr         = compute_pcr(df)
    max_pain    = compute_max_pain(df)
    max_pain_dist = round((underlying - max_pain) / max_pain * 100
                          if max_pain else 0, 3)

    # ATM IV (closest strike to underlying)
    if underlying > 0:
        atm_idx = (df["strike"] - underlying).abs().idxmin()
        atm_ce_iv = float(df.loc[atm_idx, "ce_iv"])
        atm_pe_iv = float(df.loc[atm_idx, "pe_iv"])
    else:
        atm_ce_iv = atm_pe_iv = 0.0

    # OI change totals
    ce_oi_change = float(df["ce_change_oi"].sum())
    pe_oi_change = float(df["pe_oi_change"].sum() if "pe_oi_change" in df.columns
                         else df.get("pe_change_oi", pd.Series([0])).sum())
    oi_change_pct = round(
        (pe_oi_change - ce_oi_change) / (df["ce_oi"].sum() + 1e-9) * 100, 3
    )

    if save_history:
        pcr_hist = update_pcr_history(symbol, pcr, underlying, max_pain)
        update_iv_history(symbol, atm_ce_iv, atm_pe_iv)

        # Compute IV percentile
        iv_hist = pd.read_csv(_iv_history_path(symbol),
                               index_col=0, parse_dates=True)
        iv_pct = compute_iv_percentile(iv_hist["avg_iv"])

        # PCR rolling averages
        pcr_3d = float(pcr_hist["pcr"].tail(3).mean()) if len(pcr_hist) >= 3 else pcr
        pcr_7d = float(pcr_hist["pcr"].tail(7).mean()) if len(pcr_hist) >= 7 else pcr
    else:
        iv_pct = 50.0
        pcr_3d = pcr_7d = pcr

    metrics = {
        "symbol":           symbol,
        "timestamp":        datetime.now(IST).isoformat(),
        "underlying":       underlying,
        "pcr":              pcr,
        "pcr_3d_avg":       round(pcr_3d, 4),
        "pcr_7d_avg":       round(pcr_7d, 4),
        "max_pain":         max_pain,
        "max_pain_dist_pct": max_pain_dist,
        "atm_ce_iv":        atm_ce_iv,
        "atm_pe_iv":        atm_pe_iv,
        "iv_percentile":    iv_pct,
        "total_ce_oi":      int(df["ce_oi"].sum()),
        "total_pe_oi":      int(df["pe_oi"].sum()),
        "ce_oi_change":     ce_oi_change,
        "pe_oi_change":     pe_oi_change,
        "oi_change_pct":    oi_change_pct,
    }

    log.info(
        f"{symbol}: PCR={pcr:.3f}  MaxPain={max_pain:.0f}  "
        f"MaxPainDist={max_pain_dist:.2f}%  IVPct={iv_pct:.0f}  "
        f"Underlying={underlying:.2f}"
    )
    return metrics


def update_all(save_history: bool = True) -> dict:
    """Update option chain metrics for all index instruments."""
    all_metrics = {}
    for sym in NSE_OC_SYMBOLS:
        m = update_symbol_oc(sym, save_history=save_history)
        if m:
            all_metrics[sym] = m
        time.sleep(2)  # NSE rate limit

    # Save consolidated snapshot
    path = os.path.join(OPTION_CHAIN_DIR, "latest_metrics.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    log.info(f"Option chain metrics saved → {path}")
    return all_metrics


def load_latest_metrics(symbol: str | None = None) -> dict:
    path = os.path.join(OPTION_CHAIN_DIR, "latest_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get(symbol, data) if symbol else data


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NSE Option Chain collector")
    parser.add_argument("--symbol",  default=None,
                        help="Symbol to fetch (e.g. NIFTY50, BANKNIFTY)")
    parser.add_argument("--history", action="store_true",
                        help="Save daily PCR/IV history snapshot")
    args = parser.parse_args()

    os.makedirs(OPTION_CHAIN_DIR, exist_ok=True)

    if args.symbol:
        m = update_symbol_oc(args.symbol, save_history=args.history)
        print(json.dumps(m, indent=2, default=str))
    else:
        all_m = update_all(save_history=True)
        print(json.dumps(all_m, indent=2, default=str))


if __name__ == "__main__":
    main()
