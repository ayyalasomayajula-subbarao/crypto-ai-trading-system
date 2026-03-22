"""
Paper Trader — India Stocks
Market-hours aware, F&O lot-size aware, bidirectional (LONG/SHORT).
Tracks positions, P&L, Sharpe, win rate.
"""

from __future__ import annotations
import os
import sys
import json
import logging
from datetime import datetime
from threading import RLock as Lock

import pandas as pd
import numpy as np
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INSTRUMENTS, PAPER_STATE_PATH, TIMEZONE,
    NSE_HOLIDAYS, MARKET_OPEN_TIME, MARKET_CLOSE_TIME,
)

log = logging.getLogger(__name__)
IST = pytz.timezone(TIMEZONE)

INITIAL_CAPITAL     = 1_000_000   # ₹10 lakh (₹1M)
MAX_POSITION_PCT    = 0.15        # 15% of capital per trade (before WF multiplier)
MAX_OPEN_POSITIONS  = 5
MAX_DAILY_LOSS_PCT  = 0.03        # stop trading if -3% on day
MIN_CONFIDENCE      = 0.50        # minimum signal score for entry

# ─── WF-based position sizing tiers ─────────────────────────────────────────
# Derived from fold consistency analysis (2026-03-18).
# Updated manually after each WF run; dynamic fallback handles new symbols.
# Sizing tier by valid fold count: 5+→0.7x, 3-4→0.5x, 2→0.3x, <2→0.0x
# MARGINAL_WR gate: 0.38 (vs 0.44 crypto) — daily stocks use 3:1+ R:R.
_WF_SIZE_MULTIPLIERS: dict[str, float] = {
    # ── WF run: 2026-03-22 (2010 data start, 5 folds, MARGINAL_WR_LOW=0.33, confidence spread sweep) ─
    # Tier: 5+ valid folds → 0.7x | 3-4 → 0.5x | 2 → 0.3x | NOT_VIABLE → 0.0x
    # spread= shows the best confidence_spread from hyperparameter sweep
    # ── Indices ──────────────────────────────────────────────────────────────
    "NIFTY50":    0.5,   # VIABLE   — 3 valid folds, WR=48.9%, Sharpe=1.744, spread=0.00
    "BANKNIFTY":  0.5,   # VIABLE   — 3 valid folds, WR=57.5%, Sharpe=3.341, spread=0.15
    "NIFTYIT":    0.5,   # MARGINAL — 4 valid folds, WR=39.9%, Sharpe=1.118, spread=0.05
    # ── Stocks: 5 valid folds (0.7x) ─────────────────────────────────────────
    "DRREDDY":    0.7,   # MARGINAL — 5 valid folds, WR=33.1%, Sharpe=1.171, spread=0.10
    "HCLTECH":    0.7,   # MARGINAL — 5 valid folds, WR=35.7%, Sharpe=1.387, spread=0.05
    "RELIANCE":   0.7,   # MARGINAL — 5 valid folds, WR=38.4%, Sharpe=1.711, spread=0.10
    "SBIN":       0.7,   # MARGINAL — 5 valid folds, WR=33.7%, Sharpe=1.314, spread=0.15
    "HDFCBANK":   0.7,   # MARGINAL — 5 valid folds, WR=39.6%, Sharpe=0.952, spread=0.00
    "LT":         0.7,   # MARGINAL — 5 valid folds, WR=38.7%, Sharpe=1.015, spread=0.10
    "ASIANPAINT": 0.7,   # MARGINAL — 5 valid folds, WR=33.7%, Sharpe=0.910, spread=0.10
    "TCS":        0.7,   # MARGINAL — 5 valid folds, WR=33.1%, Sharpe=0.733, spread=0.10
    "TITAN":      0.7,   # MARGINAL — 5 valid folds, WR=33.4%, Sharpe=0.728, spread=0.00
    # ── Stocks: 3-4 valid folds (0.5x) ───────────────────────────────────────
    "INFY":       0.5,   # MARGINAL — 4 valid folds, WR=35.9%, Sharpe=1.315, spread=0.15
    "MARUTI":     0.5,   # MARGINAL — 3 valid folds, WR=40.0%, Sharpe=1.500, spread=0.00
    "ULTRACEMCO": 0.5,   # MARGINAL — 4 valid folds, WR=38.4%, Sharpe=1.304, spread=0.00
    "BAJAJFINSV": 0.5,   # MARGINAL — 3 valid folds, WR=37.6%, Sharpe=1.282, spread=0.00
    "BAJFINANCE": 0.5,   # MARGINAL — 3 valid folds, WR=34.3%, Sharpe=1.217, spread=0.15
    # ── Stocks: 2 valid folds (0.3x) ─────────────────────────────────────────
    "AXISBANK":   0.3,   # MARGINAL — 2 valid folds, WR=41.6%, Sharpe=2.161, spread=0.00
    "CIPLA":      0.3,   # MARGINAL — 2 valid folds, WR=38.1%, Sharpe=1.755, spread=0.00
    "KOTAKBANK":  0.3,   # MARGINAL — 2 valid folds, WR=39.9%, Sharpe=1.530, spread=0.00
    # ── NOT_VIABLE ────────────────────────────────────────────────────────────
    "ICICIBANK":  0.0,   # NOT_VIABLE — Sharpe=1.784, WR=40.7% BUT only 1/2 folds positive (fails consistency)
    "WIPRO":      0.0,   # NOT_VIABLE — Sharpe=0.618, WR=31.6%
    "BHARTIARTL": 0.0,   # NOT_VIABLE — Sharpe=0.854, WR=32.1%
    "SUNPHARMA":  0.0,   # NOT_VIABLE — Sharpe=0.825, WR=31.2%
    "TECHM":      0.0,   # NOT_VIABLE — Sharpe=0.830, WR=28.9%
}


def _get_size_multiplier(symbol: str) -> float:
    """Return position-size multiplier for *symbol*, falling back to a dynamic
    calculation from wf_results.json for symbols not in the hardcoded table."""
    if symbol in _WF_SIZE_MULTIPLIERS:
        return _WF_SIZE_MULTIPLIERS[symbol]
    # Dynamic fallback for new symbols
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    wf_path = os.path.join(models_dir, symbol, "wf_results.json")
    if not os.path.exists(wf_path):
        return 0.0
    try:
        with open(wf_path) as f:
            wf = json.load(f)
        verdict = wf.get("verdict", "NOT_VIABLE")
        if verdict == "NOT_VIABLE":
            return 0.0
        folds = wf.get("fold_results", [])
        # Recent edge failure: last fold with ≥12 trades and 0% WR
        substantial = [f for f in folds if f.get("n_trades", 0) >= 12]
        recent_failed = bool(substantial and substantial[-1].get("wr", 0) == 0.0)
        if verdict == "VIABLE":
            if recent_failed:
                return 0.0
            freq_ratio = sum(1 for f in folds if f.get("n_trades", 0) >= 10) / max(len(folds), 1)
            return 1.0 if freq_ratio >= 0.50 else 0.5
        # MARGINAL — fold consistency (sharpe > 0 AND wr >= 40%)
        meaningful = [f for f in folds if f.get("sharpe", -99) != -99 and f.get("n_trades", 0) >= 5]
        positive   = [f for f in meaningful if f.get("sharpe", 0) > 0 and f.get("wr", 0) >= 0.40]
        ratio = len(positive) / len(meaningful) if meaningful else 0.0
        if ratio >= 0.50:
            base = 0.7
        elif ratio >= 0.25:
            base = 0.5
        else:
            base = 0.25
        if recent_failed:
            base = max(0.25, base - 0.25)
        return base
    except Exception:
        return 0.0


# ─── Market hours ────────────────────────────────────────────────────────────

def is_market_open(ts: datetime | None = None) -> bool:
    now = ts or datetime.now(IST)
    if now.weekday() >= 5:  # Saturday, Sunday
        return False
    date_str = now.strftime("%Y-%m-%d")
    if date_str in NSE_HOLIDAYS:
        return False
    open_t  = datetime.strptime(MARKET_OPEN_TIME,  "%H:%M").replace(tzinfo=IST)
    close_t = datetime.strptime(MARKET_CLOSE_TIME, "%H:%M").replace(tzinfo=IST)
    now_t   = now.replace(year=1900, month=1, day=1)
    open_t  = open_t.replace(year=1900, month=1, day=1)
    close_t = close_t.replace(year=1900, month=1, day=1)
    return open_t <= now_t <= close_t


# ─── State management ────────────────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "capital":        INITIAL_CAPITAL,
        "available":      INITIAL_CAPITAL,
        "open_positions": [],
        "closed_trades":  [],
        "daily_pnl":      0.0,
        "last_reset_date": datetime.now(IST).strftime("%Y-%m-%d"),
        "running":        False,
    }


def load_state() -> dict:
    if os.path.exists(PAPER_STATE_PATH):
        with open(PAPER_STATE_PATH) as f:
            return json.load(f)
    return _default_state()


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(PAPER_STATE_PATH), exist_ok=True)
    with open(PAPER_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ─── Paper trader class ───────────────────────────────────────────────────────

class PaperTrader:
    def __init__(self):
        self._lock  = Lock()
        self._state = load_state()
        self._reset_daily_if_needed()

    def _reset_daily_if_needed(self):
        today = datetime.now(IST).strftime("%Y-%m-%d")
        if self._state.get("last_reset_date") != today:
            self._state["daily_pnl"]      = 0.0
            self._state["last_reset_date"] = today
            save_state(self._state)

    # ── Entry ─────────────────────────────────────────────────────────────────

    def open_position(self, symbol: str, verdict: dict) -> dict:
        with self._lock:
            self._reset_daily_if_needed()

            direction = verdict.get("direction", "NEUTRAL")
            score     = verdict.get("score", 0)

            # Safety checks
            if direction == "NEUTRAL":
                return {"ok": False, "reason": "No directional signal"}

            if not is_market_open():
                return {"ok": False, "reason": "Market closed"}

            if self._state["daily_pnl"] <= -self._state["capital"] * MAX_DAILY_LOSS_PCT:
                return {"ok": False, "reason": "Daily loss limit reached"}

            open_pos = self._state["open_positions"]
            if len(open_pos) >= MAX_OPEN_POSITIONS:
                return {"ok": False, "reason": "Max positions reached"}

            if any(p["symbol"] == symbol for p in open_pos):
                return {"ok": False, "reason": f"Already in {symbol}"}

            if score / 100 < MIN_CONFIDENCE:
                return {"ok": False, "reason": f"Confidence too low ({score:.0f})"}

            wf_mult = _get_size_multiplier(symbol)
            if wf_mult == 0.0:
                return {"ok": False, "reason": f"{symbol} disabled — recent fold edge failure (0.0x tier)"}

            cfg          = INSTRUMENTS[symbol]
            lot_size     = cfg["lot_size"]
            entry_price  = verdict["entry_price"]
            tp_pct       = cfg["tp_pct"] / 100
            sl_pct       = cfg["sl_pct"] / 100

            # Position sizing: MAX_POSITION_PCT × WF multiplier
            position_value = self._state["capital"] * MAX_POSITION_PCT * wf_mult
            lots           = max(1, int(position_value / (entry_price * lot_size)))
            qty            = lots * lot_size
            trade_value    = qty * entry_price

            if trade_value > self._state["available"]:
                return {"ok": False, "reason": "Insufficient capital"}

            # Targets
            if direction == "LONG":
                target = round(entry_price * (1 + tp_pct), 2)
                sl     = round(entry_price * (1 - sl_pct), 2)
            else:
                target = round(entry_price * (1 - tp_pct), 2)
                sl     = round(entry_price * (1 + sl_pct), 2)

            position = {
                "symbol":       symbol,
                "direction":    direction,
                "entry_price":  entry_price,
                "qty":          qty,
                "lots":         lots,
                "lot_size":     lot_size,
                "trade_value":  round(trade_value, 2),
                "target_price": target,
                "sl_price":     sl,
                "entry_time":   datetime.now(IST).isoformat(),
                "unrealized_pnl": 0.0,
                "unrealized_pct": 0.0,
                "verdict_score":  score,
                "time_limit_days": cfg["time_limit_days"],
                "wf_size_mult":   wf_mult,
            }

            self._state["open_positions"].append(position)
            self._state["available"] -= trade_value
            save_state(self._state)

            log.info(
                f"[PAPER] OPEN {direction} {symbol}  "
                f"qty={qty} @ {entry_price:.2f}  "
                f"TP={target:.2f}  SL={sl:.2f}  "
                f"wf_mult={wf_mult:.1f}x"
            )
            return {"ok": True, "position": position}

    # ── Update unrealized P&L ─────────────────────────────────────────────────

    def update_prices(self, prices: dict[str, float]) -> None:
        """Call periodically with latest prices to update unrealized P&L."""
        with self._lock:
            now  = datetime.now(IST)
            to_close = []

            for pos in self._state["open_positions"]:
                sym   = pos["symbol"]
                price = prices.get(sym)
                if price is None:
                    continue

                qty   = pos["qty"]
                entry = pos["entry_price"]
                dirn  = pos["direction"]

                if dirn == "LONG":
                    raw_pnl = (price - entry) * qty
                    pct     = (price - entry) / entry
                else:
                    raw_pnl = (entry - price) * qty
                    pct     = (entry - price) / entry

                pos["unrealized_pnl"] = round(raw_pnl, 2)
                pos["unrealized_pct"] = round(pct * 100, 3)
                pos["current_price"]  = price

                # Auto-exit on SL/TP hit
                if dirn == "LONG":
                    if price >= pos["target_price"]:
                        to_close.append((sym, "TP_HIT", price))
                    elif price <= pos["sl_price"]:
                        to_close.append((sym, "SL_HIT", price))
                else:
                    if price <= pos["target_price"]:
                        to_close.append((sym, "TP_HIT", price))
                    elif price >= pos["sl_price"]:
                        to_close.append((sym, "SL_HIT", price))

                # Time limit exit
                entry_dt = datetime.fromisoformat(pos["entry_time"])
                if entry_dt.tzinfo is None:
                    entry_dt = IST.localize(entry_dt)
                days_held = (now - entry_dt).days
                if days_held >= pos["time_limit_days"]:
                    to_close.append((sym, "TIME_LIMIT", price))

            save_state(self._state)

            for sym, reason, exit_price in to_close:
                self._close_position(sym, exit_price, reason)

    # ── Close position ────────────────────────────────────────────────────────

    def _close_position(self, symbol: str, exit_price: float,
                         reason: str = "MANUAL") -> dict:
        pos_list = self._state["open_positions"]
        pos = next((p for p in pos_list if p["symbol"] == symbol), None)
        if pos is None:
            return {"ok": False, "reason": "Position not found"}

        qty   = pos["qty"]
        entry = pos["entry_price"]
        dirn  = pos["direction"]

        if dirn == "LONG":
            pnl     = (exit_price - entry) * qty
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl     = (entry - exit_price) * qty
            pnl_pct = (entry - exit_price) / entry

        trade = {
            **pos,
            "exit_price":    round(exit_price, 2),
            "exit_time":     datetime.now(IST).isoformat(),
            "exit_reason":   reason,
            "realized_pnl":  round(pnl, 2),
            "realized_pct":  round(pnl_pct * 100, 3),
            "won":           pnl > 0,
        }

        self._state["open_positions"] = [
            p for p in pos_list if p["symbol"] != symbol
        ]
        self._state["closed_trades"].append(trade)
        self._state["available"] += pos["trade_value"] + pnl
        self._state["daily_pnl"] += pnl

        save_state(self._state)

        log.info(
            f"[PAPER] CLOSE {dirn} {symbol}  "
            f"exit={exit_price:.2f}  PnL={pnl:+.2f}  "
            f"({pnl_pct:+.2%})  reason={reason}"
        )
        return {"ok": True, "trade": trade}

    def close_position(self, symbol: str, exit_price: float,
                        reason: str = "MANUAL") -> dict:
        with self._lock:
            return self._close_position(symbol, exit_price, reason)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        with self._lock:
            trades  = self._state["closed_trades"]
            capital = self._state["capital"]
            avail   = self._state["available"]
            open_pos = self._state["open_positions"]

            if not trades:
                return {
                    "total_trades": 0, "win_rate": 0, "profit_factor": 0,
                    "total_pnl": 0, "total_pnl_pct": 0, "sharpe": 0,
                    "capital": capital, "available": avail,
                    "open_positions": len(open_pos),
                    "daily_pnl": self._state["daily_pnl"],
                }

            rets    = [t["realized_pct"] / 100 for t in trades]
            wins    = [r for r in rets if r > 0]
            losses  = [r for r in rets if r <= 0]
            wr      = len(wins) / len(rets)
            pf      = sum(wins) / (abs(sum(losses)) + 1e-9)
            sharpe  = (np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(252) if len(rets) > 1 else 0
            total_pnl = sum(t["realized_pnl"] for t in trades)

            return {
                "total_trades":   len(trades),
                "win_rate":       round(wr, 3),
                "profit_factor":  round(pf, 3),
                "total_pnl":      round(total_pnl, 2),
                "total_pnl_pct":  round(total_pnl / capital * 100, 2),
                "sharpe":         round(sharpe, 3),
                "avg_win_pct":    round(np.mean(wins) * 100, 2) if wins else 0,
                "avg_loss_pct":   round(np.mean(losses) * 100, 2) if losses else 0,
                "capital":        capital,
                "available":      round(avail, 2),
                "open_positions": len(open_pos),
                "daily_pnl":      round(self._state["daily_pnl"], 2),
                "long_trades":    sum(1 for t in trades if t["direction"] == "LONG"),
                "short_trades":   sum(1 for t in trades if t["direction"] == "SHORT"),
            }

    def get_status(self) -> dict:
        with self._lock:
            return {
                "running":        self._state["running"],
                "open_positions": self._state["open_positions"],
                "metrics":        self.get_metrics(),
                "market_open":    is_market_open(),
            }

    def reset(self) -> None:
        with self._lock:
            self._state = _default_state()
            save_state(self._state)
            log.info("[PAPER] State reset")

    def start(self) -> None:
        with self._lock:
            self._state["running"] = True
            save_state(self._state)

    def stop(self) -> None:
        with self._lock:
            self._state["running"] = False
            save_state(self._state)
