"""
Feature Engineering — India Stocks
Builds multi-timeframe technical features + F&O signals + FII/DII + calendar events.
Output: ~105 features per symbol, saved to data/{SYMBOL}_features.csv

NO LOOKAHEAD: all higher-TF features use the PREVIOUS completed candle
(same floor-subtract method as crypto system).

Usage:
    python feature_engineering.py                  # all symbols
    python feature_engineering.py --symbol NIFTY50
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INSTRUMENTS, ALL_SYMBOLS, DATA_DIR,
    ohlcv_path, features_path, FII_DII_PATH,
    INDIA_VIX_PATH, OPTION_CHAIN_DIR, TIMEZONE,
    SECTOR_INDEX, EARNINGS_CAL_PATH, RBI_POLICY_PATH,
)
import pytz
IST = pytz.timezone(TIMEZONE)

# ─── Sector data cache (module-level, avoids re-downloading across symbols) ──
_sector_cache: dict = {}
SECTOR_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sector_cache")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Load helpers ────────────────────────────────────────────────────────────

def _normalize_tz(idx) -> pd.DatetimeIndex:
    """Convert any index to DatetimeIndex in Asia/Kolkata tz."""
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)  # utc=True handles mixed-tz strings
    if idx.tz is None:
        return idx.tz_localize(IST)
    return idx.tz_convert(IST)


def _load_tf(symbol: str, tf: str) -> pd.DataFrame:
    path = ohlcv_path(symbol, tf)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = _normalize_tz(df.index)
    return df.sort_index()


def _safe_shift_merge(base_df: pd.DataFrame,
                       higher_df: pd.DataFrame,
                       prefix: str,
                       cols: list[str]) -> pd.DataFrame:
    """
    Merge higher-TF features into 1H base using PREVIOUS completed candle.
    Uses asof merge — no lookahead.
    """
    higher = higher_df[cols].copy()
    # Shift by 1 candle so we use the PREVIOUS completed bar
    higher = higher.shift(1).dropna(how="all")
    higher.columns = [f"{prefix}_{c}" for c in cols]

    result = pd.merge_asof(
        base_df.reset_index(),
        higher.reset_index(),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")
    return result


# ─── Technical indicators (vectorised) ──────────────────────────────────────

def _add_sma_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    c = df["close"]
    for period in [21, 50]:
        sma = c.rolling(period, min_periods=period).mean()
        df[f"{prefix}_sma_{period}"] = sma
        df[f"{prefix}_dist_sma_{period}"] = (c - sma) / sma
        df[f"{prefix}_sma_{period}_slope"] = sma.pct_change(5)
    return df


def _add_rsi(df: pd.DataFrame, prefix: str, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    df[f"{prefix}_rsi"] = 100 - (100 / (1 + rs))
    return df


def _add_macd(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df[f"{prefix}_macd_diff"] = macd - sig
    return df


def _add_bollinger(df: pd.DataFrame, prefix: str, period: int = 20) -> pd.DataFrame:
    sma  = df["close"].rolling(period).mean()
    std  = df["close"].rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    df[f"{prefix}_bb_width"]    = (upper - lower) / (sma + 1e-9)
    df[f"{prefix}_bb_position"] = (df["close"] - lower) / (upper - lower + 1e-9)
    return df


def _add_atr(df: pd.DataFrame, prefix: str, period: int = 14) -> pd.DataFrame:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df[f"{prefix}_atr_pct"] = tr.rolling(period).mean() / (df["close"] + 1e-9)
    return df


def _add_adx(df: pd.DataFrame, prefix: str, period: int = 14) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    dm_plus  = (high - high.shift()).clip(lower=0)
    dm_minus = (low.shift() - low).clip(lower=0)
    # Zero out when not the dominant direction
    dm_plus[dm_plus < dm_minus]  = 0
    dm_minus[dm_minus < dm_plus] = 0

    hl  = high - low
    hpc = (high - close.shift()).abs()
    lpc = (low  - close.shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

    atr       = tr.ewm(span=period, adjust=False).mean()
    di_plus   = 100 * dm_plus.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    di_minus  = 100 * dm_minus.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    dx        = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    df[f"{prefix}_adx"] = dx.ewm(span=period, adjust=False).mean()
    return df


def _add_momentum(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df[f"{prefix}_momentum"] = df["close"].pct_change(10)
    return df


def _add_volume_ratio(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    vol_ma = df["volume"].rolling(20, min_periods=1).mean()
    df[f"{prefix}_volume_ratio"] = df["volume"] / (vol_ma + 1e-9)
    return df


def _add_dist_from_highs(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Distance from 52-week high/low (more meaningful than 10-period for stocks)."""
    period = 252  # trading days
    rolling_high = df["close"].rolling(period, min_periods=20).max()
    rolling_low  = df["close"].rolling(period, min_periods=20).min()
    df[f"{prefix}_dist_from_52w_high"] = (df["close"] - rolling_high) / (rolling_high + 1e-9)
    df[f"{prefix}_dist_from_52w_low"]  = (df["close"] - rolling_low)  / (rolling_low  + 1e-9)
    return df


def _add_pullback_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Distance from 63-period high — captures pullback depth from recent peak.
    For 1D timeframe this equals 1 quarter; for 1H ~2.6 days; for 4H ~10.5 days."""
    c = df["close"]
    rolling_high = c.rolling(63, min_periods=10).max()
    df[f"{prefix}_dist_from_63p_high"] = (c - rolling_high) / (rolling_high + 1e-9)
    return df


def _add_realized_vol(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    log_ret = np.log(df["close"] / df["close"].shift())
    df[f"{prefix}_realized_vol_10d"] = log_ret.rolling(10).std() * np.sqrt(252)
    df[f"{prefix}_realized_vol_30d"] = log_ret.rolling(30).std() * np.sqrt(252)
    return df


def _add_atr_percentile(df: pd.DataFrame, prefix: str, period: int = 14,
                        pct_window: int = 252) -> pd.DataFrame:
    """ATR percentile — volatility regime signal.
    Tells you whether volatility is currently high or low vs the past year.
    Near 100 = expanding vol (breakout / event-driven risk).
    Near 0   = compressed vol (consolidation, mean-reversion conditions).
    """
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    # Rolling percentile rank within the past pct_window bars
    df[f"{prefix}_atr_pct_rank"] = (
        atr.rolling(pct_window, min_periods=30)
           .rank(pct=True) * 100
    )
    return df


def build_tf_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Apply all technical indicators to a timeframe DataFrame."""
    df = df.copy()
    df = _add_sma_features(df, prefix)
    df = _add_rsi(df, prefix)
    df = _add_macd(df, prefix)
    df = _add_bollinger(df, prefix)
    df = _add_atr(df, prefix)
    df = _add_adx(df, prefix)
    df = _add_momentum(df, prefix)
    df = _add_volume_ratio(df, prefix)
    df = _add_dist_from_highs(df, prefix)
    df = _add_realized_vol(df, prefix)
    df = _add_atr_percentile(df, prefix)
    df = _add_pullback_features(df, prefix)
    return df


# ─── FII/DII features ────────────────────────────────────────────────────────

def _merge_fii_dii(base_df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(FII_DII_PATH):
        log.warning("FII/DII file missing — run collect_fii_dii.py")
        return base_df

    fii = pd.read_csv(FII_DII_PATH, index_col=0, parse_dates=True).sort_index()
    fii.index = _normalize_tz(fii.index)
    fii.index.name = "timestamp"

    cols_to_merge = [c for c in [
        "fii_net_value", "dii_net_value",
        "fii_7d_cumulative", "dii_7d_cumulative",
        "fii_dii_divergence", "fii_trend", "dii_trend",
    ] if c in fii.columns]

    if not cols_to_merge:
        return base_df

    fii_daily = fii[cols_to_merge].shift(1)  # use previous day's data (no lookahead)

    result = pd.merge_asof(
        base_df.reset_index(),
        fii_daily.reset_index(),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    # Fill weekends/holidays with last known value
    for col in cols_to_merge:
        if col in result.columns:
            result[col] = result[col].ffill()

    return result


# ─── India VIX features ──────────────────────────────────────────────────────

def _merge_india_vix(base_df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(INDIA_VIX_PATH):
        log.warning("India VIX file missing — run collect_nse_data.py")
        return base_df

    vix = pd.read_csv(INDIA_VIX_PATH, index_col=0, parse_dates=True).sort_index()
    vix.index = _normalize_tz(vix.index)
    vix.index.name = "timestamp"

    if "india_vix" not in vix.columns:
        return base_df

    vix = vix[["india_vix"]].copy()
    vix["india_vix_7d_slope"]   = vix["india_vix"].pct_change(7)
    vix["india_vix_percentile"] = (
        vix["india_vix"].rolling(252, min_periods=20).rank(pct=True) * 100
    )
    # Shifted by 1 — previous completed day, no lookahead
    vix = vix.shift(1)

    result = pd.merge_asof(
        base_df.reset_index(),
        vix.reset_index(),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    for col in ["india_vix", "india_vix_7d_slope", "india_vix_percentile"]:
        if col in result.columns:
            result[col] = result[col].ffill()

    return result


# ─── Option chain / PCR features ─────────────────────────────────────────────

def _merge_pcr(base_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Merge PCR history for index instruments."""
    pcr_path = os.path.join(OPTION_CHAIN_DIR, f"{symbol}_pcr_history.csv")
    if not os.path.exists(pcr_path):
        log.warning(f"PCR history missing for {symbol}")
        return base_df

    pcr = pd.read_csv(pcr_path, index_col=0, parse_dates=True).sort_index()
    pcr.index = _normalize_tz(pcr.index)
    pcr.index.name = "timestamp"

    pcr_feat = pd.DataFrame(index=pcr.index)
    pcr_feat["pcr"]              = pcr["pcr"]
    pcr_feat["pcr_3d_avg"]       = pcr["pcr"].rolling(3,  min_periods=1).mean()
    pcr_feat["pcr_7d_avg"]       = pcr["pcr"].rolling(7,  min_periods=1).mean()
    pcr_feat["max_pain_dist_pct"] = pcr["max_pain_dist_pct"] if "max_pain_dist_pct" in pcr.columns else 0
    pcr_feat["oi_change_pct"]    = pcr.get("oi_change_pct", pd.Series(0, index=pcr.index))
    pcr_feat = pcr_feat.shift(1)  # no lookahead

    result = pd.merge_asof(
        base_df.reset_index(),
        pcr_feat.reset_index(),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    for col in pcr_feat.columns:
        if col in result.columns:
            result[col] = result[col].ffill()

    return result


# ─── Calendar / event features ───────────────────────────────────────────────

def _add_calendar_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add event-driven features: earnings proximity, RBI dates, F&O expiry."""
    df = df.copy()
    dates = df.index.normalize()

    # F&O expiry week flag (last Thursday of month)
    def _last_thursday(ts):
        month_end = pd.Timestamp(ts.year, ts.month, 1) + pd.offsets.MonthEnd(1)
        days_back = (month_end.weekday() - 3) % 7  # 3 = Thursday
        return month_end - pd.Timedelta(days=days_back)

    def _is_expiry_week(ts):
        last_thu = _last_thursday(ts.normalize().replace(tzinfo=None))
        return abs((ts.normalize().replace(tzinfo=None) - last_thu).days) <= 4

    df["is_expiry_week"] = pd.Series(
        [int(_is_expiry_week(d)) for d in df.index], index=df.index
    )

    # Quarter end (March, June, September, December)
    df["is_quarter_end"] = df.index.month.isin([3, 6, 9, 12]).astype(int)

    # Results season (April-May, July-Aug, Oct-Nov, Jan-Feb)
    df["is_results_season"] = df.index.month.isin([4, 5, 7, 8, 10, 11, 1, 2]).astype(int)

    # Budget month (February = Union Budget)
    df["is_budget_month"] = (df.index.month == 2).astype(int)

    # RBI policy dates (bi-monthly MPC meetings — approximate)
    rbi_months = [2, 4, 6, 8, 10, 12]
    df["is_rbi_month"] = df.index.month.isin(rbi_months).astype(int)

    # Earnings proximity (stock-specific)
    if os.path.exists(EARNINGS_CAL_PATH):
        earnings = pd.read_csv(EARNINGS_CAL_PATH, parse_dates=["date"])
        sym_earnings = earnings[earnings["symbol"] == symbol]["date"].sort_values()
        days_to_earn = []
        for ts in df.index:
            future = sym_earnings[sym_earnings > ts.tz_localize(None)]
            if len(future) == 0:
                days_to_earn.append(90)
            else:
                days_to_earn.append((future.iloc[0] - ts.tz_localize(None)).days)
        df["days_to_earnings"] = days_to_earn
        df["days_to_earnings"] = df["days_to_earnings"].clip(0, 90)
    else:
        df["days_to_earnings"] = 45  # neutral default

    return df


# ─── Sector relative strength ─────────────────────────────────────────────────

_SECTOR_RS_COLS = [
    "sec_rel_ret_5d", "sector_relative_strength",
    "sec_rel_rsi", "sec_rel_dist_sma21", "sec_rel_dist_sma50", "sec_momentum",
]


def _load_sector_data(sector_name: str, sector_yf: str) -> pd.DataFrame:
    """Load sector index daily data with local CSV caching (24h TTL)."""
    global _sector_cache
    if sector_name in _sector_cache:
        return _sector_cache[sector_name]

    os.makedirs(SECTOR_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(SECTOR_CACHE_DIR, f"{sector_name}_1d.csv")

    # Use local cache if fresher than 24h
    if os.path.exists(cache_path):
        if (time.time() - os.path.getmtime(cache_path)) < 86400:
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                df.index = _normalize_tz(df.index).as_unit("us")
                df.index.name = "timestamp"
                _sector_cache[sector_name] = df
                log.info(f"Sector cache hit: {sector_name} ({len(df)} rows)")
                return df
            except Exception:
                pass  # corrupted cache — re-download below

    import yfinance as yf
    try:
        raw = yf.Ticker(sector_yf).history(period="max", interval="1d", auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        df = raw[["Close"]].rename(columns={"Close": "sec_close"})
        df.index = _normalize_tz(df.index).as_unit("us")
        df.index.name = "timestamp"
        df.to_csv(cache_path)
        log.info(f"Sector data downloaded + cached: {sector_name} ({len(df)} rows)")
        _sector_cache[sector_name] = df
        return df
    except Exception as e:
        log.warning(f"Failed to download sector data for {sector_name}: {e}")
        return pd.DataFrame()


def _compute_sector_features(sec: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features on sector close series."""
    c = sec["sec_close"]
    feat = pd.DataFrame(index=sec.index)
    feat["sec_ret_5d"]  = c.pct_change(5)
    feat["sec_ret_10d"] = c.pct_change(10)
    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    feat["sec_rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    # SMA distances
    sma21 = c.rolling(21).mean()
    sma50 = c.rolling(50).mean()
    feat["sec_dist_sma21"] = (c - sma21) / (sma21 + 1e-9)
    feat["sec_dist_sma50"] = (c - sma50) / (sma50 + 1e-9)
    return feat


def _add_sector_rs(base_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add 6 sector-relative strength features using full historical data."""
    sector    = INSTRUMENTS.get(symbol, {}).get("sector", "")
    sector_yf = SECTOR_INDEX.get(sector)
    if not sector_yf:
        for col in _SECTOR_RS_COLS:
            base_df[col] = 0.0
        return base_df

    try:
        sec = _load_sector_data(sector, sector_yf)
        if sec.empty:
            raise ValueError("empty sector data")

        # ── Stock 1D features for relative comparison ─────────────────────────
        stock_1d = _load_tf(symbol, "1d")
        if stock_1d.empty:
            raise ValueError("no 1D OHLCV for stock")

        c = stock_1d["close"]
        # Normalize precision to microseconds to match sector data index
        stock_idx = stock_1d.index.as_unit("us")
        stock_idx.name = "timestamp"
        stock_feat = pd.DataFrame(index=stock_idx)
        stock_feat["stock_ret_5d"]  = c.pct_change(5)
        stock_feat["stock_ret_10d"] = c.pct_change(10)
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        stock_feat["stock_rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
        sma21 = c.rolling(21).mean()
        sma50 = c.rolling(50).mean()
        stock_feat["stock_dist_sma21"] = (c - sma21) / (sma21 + 1e-9)
        stock_feat["stock_dist_sma50"] = (c - sma50) / (sma50 + 1e-9)

        # ── Sector features ───────────────────────────────────────────────────
        sec_feat = _compute_sector_features(sec)

        # ── Combine at daily resolution ───────────────────────────────────────
        combined = pd.merge_asof(
            stock_feat.sort_index().reset_index(),
            sec_feat.sort_index().reset_index(),
            on="timestamp",
            direction="backward",
        ).set_index("timestamp")

        # ── 6 relative features (shift=1 → no lookahead) ─────────────────────
        rel = pd.DataFrame(index=combined.index)
        rel.index.name = "timestamp"
        rel["sec_rel_ret_5d"]          = combined["stock_ret_5d"]  - combined["sec_ret_5d"]
        rel["sector_relative_strength"] = combined["stock_ret_10d"] - combined["sec_ret_10d"]
        rel["sec_rel_rsi"]             = combined["stock_rsi"]      - combined["sec_rsi"]
        rel["sec_rel_dist_sma21"]      = combined["stock_dist_sma21"] - combined["sec_dist_sma21"]
        rel["sec_rel_dist_sma50"]      = combined["stock_dist_sma50"] - combined["sec_dist_sma50"]
        rel["sec_momentum"]            = combined["sec_ret_10d"]   # sector absolute momentum

        rel = rel.shift(1)  # no lookahead: use PREVIOUS day's relative values

        # ── Merge daily relative features to 1H base ──────────────────────────
        result = pd.merge_asof(
            base_df.reset_index().sort_values("timestamp"),
            rel.reset_index().sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        ).set_index("timestamp")

        for col in _SECTOR_RS_COLS:
            if col in result.columns:
                result[col] = result[col].ffill()

        return result

    except Exception as e:
        log.warning(f"Sector RS for {symbol}: {e}")
        for col in _SECTOR_RS_COLS:
            base_df[col] = 0.0
        return base_df


# ─── Advance/Decline merge ────────────────────────────────────────────────────

def _merge_advance_decline(base_df: pd.DataFrame) -> pd.DataFrame:
    ad_path = os.path.join(DATA_DIR, "advance_decline.csv")
    if not os.path.exists(ad_path):
        return base_df

    ad = pd.read_csv(ad_path, index_col=0, parse_dates=True).sort_index()
    ad.index = _normalize_tz(ad.index)
    ad.index.name = "timestamp"

    if "ad_ratio" not in ad.columns:
        return base_df

    ad_feat = ad[["ad_ratio"]].copy()
    ad_feat["ad_ratio_7d_avg"] = ad_feat["ad_ratio"].rolling(7, min_periods=1).mean()
    ad_feat = ad_feat.shift(1)

    result = pd.merge_asof(
        base_df.reset_index(),
        ad_feat.reset_index(),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    for col in ["ad_ratio", "ad_ratio_7d_avg"]:
        if col in result.columns:
            result[col] = result[col].ffill()

    return result


# ─── GIFT Nifty proxy merge ───────────────────────────────────────────────────

def _merge_gift_proxy(base_df: pd.DataFrame) -> pd.DataFrame:
    gift_path = os.path.join(DATA_DIR, "gift_nifty_proxy.csv")
    if not os.path.exists(gift_path):
        return base_df

    gift = pd.read_csv(gift_path, index_col=0, parse_dates=True).sort_index()
    gift.index = _normalize_tz(gift.index)
    gift.index.name = "timestamp"

    if "gift_proxy_return" not in gift.columns:
        return base_df

    gift_feat = gift[["gift_proxy_return"]].shift(1)  # overnight = previous session

    result = pd.merge_asof(
        base_df.reset_index(),
        gift_feat.reset_index(),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    if "gift_proxy_return" in result.columns:
        result["gift_proxy_return"] = result["gift_proxy_return"].ffill()

    return result


# ─── Master feature builder ───────────────────────────────────────────────────

_MIN_EXTRA_DAILY_DAYS = 730   # switch to 1D base if daily has ≥730 more days than 1H


def build_features(symbol: str) -> pd.DataFrame:
    """
    Build full feature matrix for one symbol.

    Base timeframe selection:
      - 1H  → used when 1H data exists AND daily history is not much deeper
      - 1D  → used when (a) no 1H data exists, or (b) 1D data starts ≥730 days
                earlier than 1H, giving proper multi-regime WF coverage.
    """
    log.info(f"Building features for {symbol}...")

    # ── Load base timeframe ───────────────────────────────────────────────────
    df_1h = _load_tf(symbol, "1h")
    df_1d = _load_tf(symbol, "1d")

    if df_1h.empty and df_1d.empty:
        log.error(f"{symbol}: no 1H or 1D data found")
        return pd.DataFrame()

    # Prefer 1D base when daily history is significantly longer than hourly.
    # This gives stocks like MARUTI/HDFCBANK/ICICIBANK 11 years of training data
    # instead of being capped at yfinance's ~730-day 1H limit.
    if not df_1h.empty and not df_1d.empty:
        extra_days = (df_1h.index[0] - df_1d.index[0]).days
        if extra_days >= _MIN_EXTRA_DAILY_DAYS:
            log.info(
                f"{symbol}: 1D data starts {extra_days}d earlier than 1H "
                f"({df_1d.index[0].date()} vs {df_1h.index[0].date()}) — "
                f"using 1D as base for full market-cycle WF coverage"
            )
            base_tf = "1d"
            base_data = df_1d
        else:
            base_tf = "1h"
            base_data = df_1h
    elif df_1h.empty:
        base_tf = "1d"
        base_data = df_1d
        log.info(f"{symbol}: no 1H data — using 1D as base timeframe")
    else:
        base_tf = "1h"
        base_data = df_1h

    # ── Base-TF technical features ────────────────────────────────────────────
    df = build_tf_features(base_data, base_tf)

    # ── Multi-timeframe features (no lookahead via shift) ────────────────────
    # 1D base: merge only 1W (true higher TF).
    #   Adding 1H/4H to 1D rows creates 80% NaN columns (yfinance 1H limit = 2yr)
    #   → LightGBM learns "NaN = pre-2024 regime" spurious date-awareness.
    #   Solution: keep 1D+1W clean for 10yr training stability.
    #   1H/4H signals can be used at inference time as a separate entry filter.
    # 1H base: merge 4H, 1D, 1W as before — all available for full period.
    if base_tf == "1d":
        higher_tfs = [("1w", "1w")]
    else:
        higher_tfs = [("4h", "4h"), ("1d", "1d"), ("1w", "1w")]
    for tf, prefix in higher_tfs:
        df_tf = _load_tf(symbol, tf)
        if df_tf.empty:
            log.warning(f"{symbol}: missing {tf} data, skipping")
            continue
        df_tf_feat = build_tf_features(df_tf, prefix)
        # Extract feature columns (exclude OHLCV)
        feat_cols = [c for c in df_tf_feat.columns
                     if c not in ("open", "high", "low", "close", "volume")]
        df = _safe_shift_merge(df, df_tf_feat, prefix="", cols=feat_cols)

    # ── Market-wide features (same for all instruments) ───────────────────────
    df = _merge_fii_dii(df)
    df = _merge_india_vix(df)
    df = _merge_advance_decline(df)
    df = _merge_gift_proxy(df)

    # ── Index-specific F&O features ───────────────────────────────────────────
    if INSTRUMENTS.get(symbol, {}).get("type") == "index":
        df = _merge_pcr(df, symbol)

    # ── Calendar / event features ─────────────────────────────────────────────
    df = _add_calendar_features(df, symbol)

    # ── Sector relative strength (stock-specific) ────────────────────────────
    if INSTRUMENTS.get(symbol, {}).get("type") == "stock":
        df = _add_sector_rs(df, symbol)

    # ── Clean up ─────────────────────────────────────────────────────────────
    # Drop raw OHLCV columns (keep only features)
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    df = df.drop(columns=[c for c in ohlcv_cols if c in df.columns])

    # Remove duplicate columns that crept in from merge
    df = df.loc[:, ~df.columns.duplicated()]

    # Drop rows before we have enough data for most rolling windows
    df = df.iloc[252:]  # skip first 252 rows (1Y of 1H data)

    # Drop rows with >50% NaN (early data)
    threshold = int(df.shape[1] * 0.5)
    df = df.dropna(thresh=threshold)

    log.info(f"{symbol}: {df.shape[0]} rows × {df.shape[1]} features")
    return df


def update_symbol(symbol: str) -> bool:
    df = build_features(symbol)
    if df.empty:
        return False
    path = features_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    log.info(f"{symbol}: features saved → {path}")
    return True


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature engineering for India stocks")
    parser.add_argument("--symbol", default=None)
    args = parser.parse_args()

    _active_env = os.environ.get("ACTIVE_SYMBOLS", "")
    _active = [s.strip() for s in _active_env.split(",") if s.strip()] if _active_env else ALL_SYMBOLS
    symbols = [args.symbol] if args.symbol else _active
    for sym in symbols:
        update_symbol(sym)

    log.info("Feature engineering complete.")


if __name__ == "__main__":
    main()
