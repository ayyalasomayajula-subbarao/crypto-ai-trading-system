"""
Precision Verdict Engine — India Stocks
Fuses ML model + 13 market signals into a single actionable verdict.
Verdicts: STRONG_BUY / BUY / LEAN_BUY / HOLD / LEAN_SELL / SELL / STRONG_SELL / AVOID

5-minute cache. SQLite accuracy tracking.

Usage (standalone):
    from precision_verdict import VerdictEngine
    engine = VerdictEngine()
    result = engine.get_verdict("NIFTY50")
"""

from __future__ import annotations
import os
import sys
import json
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from threading import Lock

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INSTRUMENTS, MODELS_DIR, DATA_DIR,
    features_path, model_path, features_list_path,
    SIGNAL_WEIGHTS, VERDICT_THRESHOLDS, TIMEZONE,
    PCR_STRONG_BULLISH, PCR_BULLISH, PCR_NEUTRAL_LOW,
    PCR_NEUTRAL_HIGH, PCR_BEARISH, PCR_STRONG_BEARISH,
    VIX_EXTREME_FEAR, VIX_HIGH_FEAR, VIX_NEUTRAL_HIGH,
    VIX_NEUTRAL_LOW, VIX_COMPLACENCY,
    AGENT_DB_PATH, ADX_GATE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

import pytz
IST = pytz.timezone(TIMEZONE)

CACHE_TTL = 300  # 5 minutes


# ─── SQLite accuracy store ───────────────────────────────────────────────────

def _init_db():
    os.makedirs(os.path.dirname(AGENT_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(AGENT_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_verdicts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, timestamp TEXT, verdict TEXT,
            score REAL, direction TEXT,
            entry_price REAL, target_price REAL, sl_price REAL,
            resolved INTEGER DEFAULT 0,
            outcome TEXT, actual_return REAL
        )
    """)
    conn.commit()
    conn.close()


def _save_verdict(symbol: str, result: dict):
    try:
        conn = sqlite3.connect(AGENT_DB_PATH)
        conn.execute("""
            INSERT INTO stock_verdicts
            (symbol, timestamp, verdict, score, direction,
             entry_price, target_price, sl_price)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            symbol,
            result["timestamp"],
            result["verdict"],
            result["score"],
            result.get("direction", ""),
            result.get("entry_price", 0),
            result.get("target_price", 0),
            result.get("sl_price", 0),
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"DB save failed: {e}")


# ─── Model loader ─────────────────────────────────────────────────────────────

_model_cache: dict = {}
_model_lock = Lock()


def _load_model(symbol: str):
    with _model_lock:
        if symbol in _model_cache:
            return _model_cache[symbol]

        # Try WF model first, fall back to standard
        for name in ["wf_decision_model_v2.pkl", "decision_model_v2.pkl"]:
            path = model_path(symbol, name)
            if os.path.exists(path):
                import joblib
                m = joblib.load(path)
                _model_cache[symbol] = m
                log.info(f"Loaded model: {symbol}/{name}")
                return m

    log.warning(f"No model found for {symbol}")
    return None


def _load_feature_list(symbol: str) -> list[str]:
    path = features_list_path(symbol)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ─── Feature row for latest candle ────────────────────────────────────────────

def _get_latest_features(symbol: str) -> pd.Series | None:
    feat_p = features_path(symbol)
    if not os.path.exists(feat_p):
        log.warning(f"{symbol}: features file not found")
        return None

    df = pd.read_csv(feat_p, index_col=0, parse_dates=True)
    if df.empty:
        return None

    row = df.iloc[-1].copy()
    return row


def _get_current_price(symbol: str) -> float:
    """Get latest close price from 1D data."""
    from config import ohlcv_path
    path = ohlcv_path(symbol, "1d")
    if not os.path.exists(path):
        return 0.0
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        return 0.0
    return float(df["close"].iloc[-1])


# ─── Individual signal scorers ────────────────────────────────────────────────

def _score_ml_model(probas: np.ndarray, classes: list,
                    threshold: float, confidence_spread: float = 0.0) -> dict:
    """Score ML model output. Returns signal in [-1, +1] range.

    confidence_spread: require P(direction) - P(opposite) >= spread.
    Filters weak directional signals where the model is uncertain.
    0.0 = disabled (current behaviour).
    """
    up_idx   = classes.index("UP")   if "UP"   in classes else 2
    down_idx = classes.index("DOWN") if "DOWN" in classes else 0

    p_up   = float(probas[up_idx])
    p_down = float(probas[down_idx])

    if p_up > p_down and p_up >= threshold:
        # Apply confidence spread filter
        if confidence_spread > 0.0 and (p_up - p_down) < confidence_spread:
            return {"score": 0, "direction": "NEUTRAL", "p_up": p_up, "p_down": p_down}
        score = (p_up - 0.33) / 0.67  # normalize to [0,1]
        direction = "LONG"
    elif p_down > p_up and p_down >= threshold:
        if confidence_spread > 0.0 and (p_down - p_up) < confidence_spread:
            return {"score": 0, "direction": "NEUTRAL", "p_up": p_up, "p_down": p_down}
        score = -(p_down - 0.33) / 0.67
        direction = "SHORT"
    else:
        score = 0
        direction = "NEUTRAL"

    return {"score": score, "direction": direction,
            "p_up": p_up, "p_down": p_down}


def _score_sma(row: pd.Series, tf: str, period: int) -> float:
    col = f"{tf}_dist_sma_{period}"
    val = float(row.get(col, 0) or 0)
    # dist > +2% = bullish, < -2% = bearish, scale to [-1,1]
    return np.clip(val / 0.05, -1, 1)


def _score_rsi(row: pd.Series, tf: str = "1d") -> float:
    rsi = float(row.get(f"{tf}_rsi", 50) or 50)
    # < 30 = oversold (bullish), > 70 = overbought (bearish)
    if rsi < 30:   return (30 - rsi) / 30
    if rsi > 70:   return -(rsi - 70) / 30
    return (50 - rsi) / 50 * 0.3  # mild signal in neutral zone


def _score_macd(row: pd.Series, tf: str = "1d") -> float:
    val = float(row.get(f"{tf}_macd_diff", 0) or 0)
    # Normalize by ATR
    atr = float(row.get(f"{tf}_atr_pct", 0.01) or 0.01)
    return np.clip(val / (atr * 100 + 1e-9), -1, 1)


def _score_adx(row: pd.Series, tf: str = "1d") -> float:
    """ADX confirms trend strength (directional, not directional bias)."""
    adx = float(row.get(f"{tf}_adx", 0) or 0)
    if adx < ADX_GATE:
        return 0  # no trend, neutral
    # Check SMA direction to determine sign
    dist = float(row.get(f"{tf}_dist_sma_21", 0) or 0)
    sign = 1 if dist > 0 else -1
    return sign * min((adx - ADX_GATE) / 30, 1)


def _score_pcr(row: pd.Series) -> float:
    pcr = float(row.get("pcr_7d_avg", row.get("pcr", 1.0)) or 1.0)
    if pcr <= PCR_STRONG_BULLISH:   return  1.0
    if pcr <= PCR_BULLISH:          return  0.6
    if pcr <= PCR_NEUTRAL_LOW:      return  0.2
    if pcr <= PCR_NEUTRAL_HIGH:     return  0.0
    if pcr <= PCR_BEARISH:          return -0.4
    if pcr <= PCR_STRONG_BEARISH:   return -0.8
    return -1.0


def _score_india_vix(row: pd.Series) -> float:
    vix = float(row.get("india_vix", 16) or 16)
    vix_slope = float(row.get("india_vix_7d_slope", 0) or 0)
    if vix >= VIX_EXTREME_FEAR:   base = -1.0
    elif vix >= VIX_HIGH_FEAR:    base = -0.6
    elif vix <= VIX_COMPLACENCY:  base = -0.3  # complacency = risky
    elif vix <= VIX_NEUTRAL_LOW:  base =  0.4
    else:                          base =  0.0
    # VIX rising = more bearish
    slope_adj = np.clip(-vix_slope * 5, -0.3, 0.3)
    return np.clip(base + slope_adj, -1, 1)


def _score_fii(row: pd.Series) -> float:
    fii_7d = float(row.get("fii_7d_cumulative", 0) or 0)
    # Normalize by typical range (~10,000 crores in 7 days)
    return np.clip(fii_7d / 10000, -1, 1)


def _score_dii(row: pd.Series) -> float:
    dii_7d = float(row.get("dii_7d_cumulative", 0) or 0)
    return np.clip(dii_7d / 8000, -1, 1)


def _score_oi_change(row: pd.Series) -> float:
    oi_chg = float(row.get("oi_change_pct", 0) or 0)
    # Rising PE OI = bearish; rising CE OI = bullish
    return np.clip(oi_chg / 10, -1, 1)


def _score_delivery(row: pd.Series) -> float:
    """Delivery % > 50% = genuine buying."""
    del_pct = float(row.get("delivery_pct", 50) or 50)
    if del_pct >= 60:   return  0.8
    if del_pct >= 50:   return  0.4
    if del_pct >= 40:   return  0.0
    return -0.4


def _score_advance_decline(row: pd.Series) -> float:
    ad = float(row.get("ad_ratio", 0.5) or 0.5)
    # ad_ratio: fraction of advances (0.5 = neutral)
    return np.clip((ad - 0.5) * 4, -1, 1)


def _score_gift_nifty(row: pd.Series) -> float:
    gift = float(row.get("gift_proxy_return", 0) or 0)
    return np.clip(gift * 20, -1, 1)


# ─── Score aggregation ────────────────────────────────────────────────────────

def _aggregate_scores(signals: dict) -> float:
    """Weighted average of all signal scores → final score [0, 100]."""
    weights = SIGNAL_WEIGHTS
    total_w = 0
    weighted_sum = 0

    for sig_name, sig_val in signals.items():
        w = weights.get(sig_name, 1.0)
        weighted_sum += sig_val * w
        total_w += w

    raw = weighted_sum / (total_w + 1e-9)  # [-1, +1]
    return round((raw + 1) / 2 * 100, 1)   # → [0, 100]


def _score_to_verdict(score: float) -> str:
    if score >= VERDICT_THRESHOLDS["STRONG_BUY"]:    return "STRONG_BUY"
    if score >= VERDICT_THRESHOLDS["BUY"]:            return "BUY"
    if score >= VERDICT_THRESHOLDS["LEAN_BUY"]:       return "LEAN_BUY"
    if score >= VERDICT_THRESHOLDS["HOLD"]:           return "HOLD"
    if score >= VERDICT_THRESHOLDS["LEAN_SELL"]:      return "LEAN_SELL"
    if score >= VERDICT_THRESHOLDS["SELL"]:           return "SELL"
    if score >= VERDICT_THRESHOLDS["STRONG_SELL"]:    return "STRONG_SELL"
    return "AVOID"


def _compute_targets(price: float, verdict: str, cfg: dict) -> dict:
    tp = cfg["tp_pct"] / 100
    sl = cfg["sl_pct"] / 100
    direction = "LONG" if "BUY" in verdict else "SHORT" if "SELL" in verdict else "NEUTRAL"

    if direction == "LONG":
        return {
            "direction":     "LONG",
            "entry_price":   round(price, 2),
            "target_price":  round(price * (1 + tp), 2),
            "sl_price":      round(price * (1 - sl), 2),
        }
    if direction == "SHORT":
        return {
            "direction":     "SHORT",
            "entry_price":   round(price, 2),
            "target_price":  round(price * (1 - tp), 2),
            "sl_price":      round(price * (1 + sl), 2),
        }
    return {"direction": "NEUTRAL", "entry_price": price,
            "target_price": price, "sl_price": price}


# ─── Main VerdictEngine ───────────────────────────────────────────────────────

class VerdictEngine:
    def __init__(self):
        _init_db()
        self._cache: dict[str, dict] = {}
        self._cache_ts: dict[str, float] = {}
        self._lock = Lock()
        self._feat_cache: dict[str, object] = {}
        self._feat_cache_ts: dict[str, float] = {}
        # Models are lazy-loaded on first use to conserve RAM (EC2 free tier)

    def _preload_models(self):
        """Load all available models into memory so first scan is instant."""
        from concurrent.futures import ThreadPoolExecutor
        def _load(sym):
            try:
                _load_model(sym)
            except Exception:
                pass
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(_load, INSTRUMENTS.keys()))
        log.info(f"Pre-loaded {len(_model_cache)} models into cache")

    def get_verdict(self, symbol: str, force: bool = False) -> dict:
        # Quick cache check (lock only for dict access)
        now = time.time()
        with self._lock:
            if (not force and symbol in self._cache
                    and now - self._cache_ts.get(symbol, 0) < CACHE_TTL):
                return self._cache[symbol]

        # Compute outside lock — allows concurrent symbol evaluation
        result = self._compute(symbol)

        with self._lock:
            self._cache[symbol] = result
            self._cache_ts[symbol] = time.time()
        return result

    def _get_cached_features(self, symbol: str):
        """Return latest feature row, cached for CACHE_TTL seconds."""
        now = time.time()
        if (symbol in self._feat_cache
                and now - self._feat_cache_ts.get(symbol, 0) < CACHE_TTL):
            return self._feat_cache[symbol]
        row = _get_latest_features(symbol)
        self._feat_cache[symbol] = row
        self._feat_cache_ts[symbol] = now
        return row

    def _compute(self, symbol: str) -> dict:
        cfg = INSTRUMENTS.get(symbol)
        if not cfg:
            return {"symbol": symbol, "verdict": "AVOID", "score": 0,
                    "error": "Unknown symbol"}

        row = self._get_cached_features(symbol)
        if row is None:
            return {"symbol": symbol, "verdict": "AWAIT_DATA", "score": 50,
                    "error": "No features available"}

        model = _load_model(symbol)
        feature_list = _load_feature_list(symbol)
        current_price = _get_current_price(symbol)

        # ── ML signal ─────────────────────────────────────────────────────────
        ml_result = {"score": 0, "direction": "NEUTRAL", "p_up": 0.33, "p_down": 0.33}
        if model and feature_list:
            try:
                feat_vals = row.reindex(feature_list).fillna(0).values.reshape(1, -1)
                probas = model.predict_proba(feat_vals)[0]
                classes = list(model.classes_) if hasattr(model, "classes_") else ["DOWN", "SIDEWAYS", "UP"]
                # For CalibratedClassifierCV, get classes from base
                try:
                    from sklearn.calibration import CalibratedClassifierCV
                    classes = list(model.calibrated_classifiers_[0].estimator.classes_)
                except Exception:
                    pass
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.classes_ = np.array(classes if isinstance(classes[0], str) else
                                        ["DOWN", "SIDEWAYS", "UP"])
                # Load WF threshold + confidence spread from results
                threshold = 0.50
                confidence_spread = 0.0
                results_path = os.path.join(MODELS_DIR, symbol, "wf_results.json")
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        wf = json.load(f)
                    threshold = wf.get("best_threshold", 0.50) or 0.50
                    confidence_spread = wf.get("best_spread", 0.0) or 0.0
                ml_result = _score_ml_model(
                    probas, le.classes_.tolist(), threshold, confidence_spread
                )
            except Exception as e:
                log.warning(f"{symbol} ML score failed: {e}")

        # ── All signals ───────────────────────────────────────────────────────
        signals = {
            "ml_model":          ml_result["score"],
            "price_vs_sma21_1d": _score_sma(row, "1d", 21),
            "price_vs_sma50_1d": _score_sma(row, "1d", 50),
            "rsi_1d":            _score_rsi(row),
            "macd_1d":           _score_macd(row),
            "adx_1d":            _score_adx(row),
            "pcr":               _score_pcr(row),
            "india_vix":         _score_india_vix(row),
            "fii_net":           _score_fii(row),
            "dii_net":           _score_dii(row),
            "oi_change":         _score_oi_change(row),
            "delivery_pct":      _score_delivery(row),
            "advance_decline":   _score_advance_decline(row),
            "gift_nifty":        _score_gift_nifty(row),
        }

        score   = _aggregate_scores(signals)
        verdict = _score_to_verdict(score)
        targets = _compute_targets(current_price, verdict, cfg)

        # Build signal breakdown for UI
        signal_details = []
        for sig, val in signals.items():
            w = SIGNAL_WEIGHTS.get(sig, 1.0)
            signal_details.append({
                "name":       sig,
                "value":      round(val, 3),
                "weight":     w,
                "bullish":    bool(val > 0.1),
                "bearish":    bool(val < -0.1),
            })

        result = {
            "symbol":          symbol,
            "display_name":    cfg.get("display_name", symbol),
            "verdict":         verdict,
            "score":           score,
            "direction":       targets["direction"],
            "entry_price":     targets["entry_price"],
            "target_price":    targets["target_price"],
            "sl_price":        targets["sl_price"],
            "current_price":   current_price,
            "signals":         signal_details,
            "ml_p_up":         round(ml_result.get("p_up", 0), 3),
            "ml_p_down":       round(ml_result.get("p_down", 0), 3),
            "timestamp":       datetime.now(IST).isoformat(),
            "tp_pct":          cfg["tp_pct"],
            "sl_pct":          cfg["sl_pct"],
        }

        _save_verdict(symbol, result)
        self._check_accuracy(symbol)
        return result

    def _check_accuracy(self, symbol: str) -> None:
        """Resolve old unresolved verdicts by checking if TP/SL was hit using
        daily close prices. Uses high/low columns when available for better
        TP/SL detection (daily candle may have touched target intraday)."""
        try:
            from config import ohlcv_path
            path = ohlcv_path(symbol, "1d")
            if not os.path.exists(path):
                return
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if df.empty:
                return

            cfg          = INSTRUMENTS[symbol]
            time_limit   = cfg.get("time_limit_days", 7)
            now          = datetime.now(IST)

            conn = sqlite3.connect(AGENT_DB_PATH)
            rows = conn.execute(
                """SELECT id, timestamp, verdict, direction, entry_price,
                          target_price, sl_price
                   FROM stock_verdicts
                   WHERE symbol=? AND resolved=0
                   ORDER BY id DESC LIMIT 30""",
                (symbol,)
            ).fetchall()

            for row in rows:
                rid, ts_str, verdict, direction, entry, target, sl = row
                if not entry or entry <= 0:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = IST.localize(ts)
                except Exception:
                    continue

                days_elapsed = (now - ts).total_seconds() / 86400
                # Only resolve after at least 1 full trading day
                if days_elapsed < 1:
                    continue

                # Get candles since verdict date (keep tz-aware for comparison)
                future_df = df[df.index > pd.Timestamp(ts)]
                if future_df.empty:
                    continue

                outcome       = None
                actual_return = None

                if direction in ("LONG", "SHORT"):
                    for _, candle in future_df.iterrows():
                        hi  = float(candle.get("high",  candle["close"]))
                        lo  = float(candle.get("low",   candle["close"]))
                        cls = float(candle["close"])

                        if direction == "LONG":
                            if hi >= target:
                                outcome       = "WIN"
                                actual_return = round((target - entry) / entry * 100, 2)
                                break
                            if lo <= sl:
                                outcome       = "LOSS"
                                actual_return = round((sl - entry) / entry * 100, 2)
                                break
                        else:  # SHORT
                            if lo <= target:
                                outcome       = "WIN"
                                actual_return = round((entry - target) / entry * 100, 2)
                                break
                            if hi >= sl:
                                outcome       = "LOSS"
                                actual_return = round((entry - sl) / entry * 100, 2)
                                break

                    # Time limit exit — use last available close
                    if outcome is None and days_elapsed >= time_limit:
                        last_close    = float(future_df["close"].iloc[-1])
                        if direction == "LONG":
                            actual_return = round((last_close - entry) / entry * 100, 2)
                        else:
                            actual_return = round((entry - last_close) / entry * 100, 2)
                        outcome = "WIN" if actual_return > 0 else "LOSS"
                else:
                    # NEUTRAL / HOLD verdict — mark resolved after time_limit
                    if days_elapsed >= time_limit:
                        outcome       = "NEUTRAL"
                        actual_return = 0.0

                if outcome is not None:
                    conn.execute(
                        """UPDATE stock_verdicts
                           SET resolved=1, outcome=?, actual_return=?
                           WHERE id=?""",
                        (outcome, actual_return, rid)
                    )

            conn.commit()
            conn.close()
        except Exception as e:
            log.warning(f"Accuracy check failed for {symbol}: {e}")

    def get_accuracy_stats(self, symbol: str, days: int = 30) -> dict:
        """Return verdict accuracy stats for symbol over last N days."""
        try:
            cutoff = (datetime.now(IST) - timedelta(days=days)).isoformat()
            conn   = sqlite3.connect(AGENT_DB_PATH)
            rows   = conn.execute(
                """SELECT verdict, outcome, actual_return
                   FROM stock_verdicts
                   WHERE symbol=? AND resolved=1 AND timestamp >= ?
                   ORDER BY timestamp DESC""",
                (symbol, cutoff)
            ).fetchall()
            conn.close()

            if not rows:
                return {"symbol": symbol, "total": 0, "win_rate": 0,
                        "avg_return": 0, "by_verdict": {}, "days": days}

            total  = len(rows)
            wins   = sum(1 for _, o, _ in rows if o == "WIN")
            rets   = [r for _, _, r in rows if r is not None]
            by_v: dict = {}
            for v, o, r in rows:
                if v not in by_v:
                    by_v[v] = {"total": 0, "wins": 0, "avg_return": 0, "_rets": []}
                by_v[v]["total"] += 1
                if o == "WIN":
                    by_v[v]["wins"] += 1
                if r is not None:
                    by_v[v]["_rets"].append(r)
            for v in by_v:
                _r = by_v[v].pop("_rets")
                by_v[v]["avg_return"] = round(sum(_r) / len(_r), 2) if _r else 0
                by_v[v]["win_rate"]   = round(by_v[v]["wins"] / by_v[v]["total"] * 100, 1)

            return {
                "symbol":     symbol,
                "total":      total,
                "wins":       wins,
                "win_rate":   round(wins / total * 100, 1),
                "avg_return": round(sum(rets) / len(rets), 2) if rets else 0,
                "by_verdict": by_v,
                "days":       days,
            }
        except Exception as e:
            log.warning(f"Accuracy stats failed for {symbol}: {e}")
            return {"symbol": symbol, "total": 0, "win_rate": 0,
                    "avg_return": 0, "by_verdict": {}, "days": days}

    def get_verdict_history(self, symbol: str, limit: int = 50) -> list[dict]:
        try:
            conn = sqlite3.connect(AGENT_DB_PATH)
            rows = conn.execute(
                """SELECT timestamp, verdict, score, direction,
                          entry_price, target_price, sl_price,
                          outcome, actual_return
                   FROM stock_verdicts
                   WHERE symbol=? ORDER BY timestamp DESC LIMIT ?""",
                (symbol, limit)
            ).fetchall()
            conn.close()
            return [
                dict(zip(["timestamp","verdict","score","direction",
                          "entry_price","target_price","sl_price",
                          "outcome","actual_return"], r))
                for r in rows
            ]
        except Exception as e:
            log.warning(f"History fetch failed: {e}")
            return []

    def scan_all(self) -> list[dict]:
        """Scan active symbols and return ranked verdicts."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from config import SCAN_SYMBOLS
        results = []
        with ThreadPoolExecutor(max_workers=min(6, len(SCAN_SYMBOLS))) as ex:
            futures = {ex.submit(self.get_verdict, sym): sym
                       for sym in SCAN_SYMBOLS}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    log.error(f"Scan failed for {futures[fut]}: {e}")
        return sorted(results, key=lambda x: x["score"], reverse=True)
