"""
Walk-Forward Validation — India Stocks
Expanding window WF (same methodology as crypto system):
  - Train anchored to data start, grows each fold
  - Bidirectional: LONG on UP signal, SHORT on DOWN signal
  - Regime gate: ADX >= 20 + SMA-50 weekly regime
  - Meta-labeling: optional secondary model on last 25% of train window
  - Saves wf_decision_model_v2.pkl from the most recent fold

Usage:
    python walk_forward.py                        # all symbols (parallel)
    python walk_forward.py --symbol NIFTY50
    python walk_forward.py --fast                 # 3-5x faster, lighter params
    python walk_forward.py --workers 6            # parallelism level
"""

from __future__ import annotations
import os
import sys
import json
import logging
import argparse
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import pytz

_IST = pytz.timezone("Asia/Kolkata")

def _ts(date_str: str) -> pd.Timestamp:
    """Convert date string to tz-aware Timestamp in IST for index comparisons."""
    return pd.Timestamp(date_str).tz_localize(_IST)

def _norm_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize any datetime index to Asia/Kolkata."""
    if idx.tz is None:
        return idx.tz_localize(_IST)
    return idx.tz_convert(_IST)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INSTRUMENTS, ALL_SYMBOLS, MODELS_DIR,
    features_path, model_path, features_list_path,
    LGBM_PARAMS, META_LGBM_PARAMS, WF_FAST_LGBM_PARAMS,
    WF_THRESHOLD_GRID, META_LABELING, META_WIN_THRESH, META_TRAIN_FRAC,
    VIABLE_SHARPE, VIABLE_WR, MARGINAL_SHARPE, MARGINAL_WR,
    MARGINAL_SHARPE_HIGH, MARGINAL_WR_LOW, MIN_TRADES,
    ADX_GATE,
)

# Set by --fast flag at runtime; affects WF sweep only (final model always uses LGBM_PARAMS)
_WF_PARAMS = LGBM_PARAMS
_WF_CV     = 3
from train_model import _load_features_and_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Fold definitions ─────────────────────────────────────────────────────────

def _build_folds(X: pd.DataFrame, n_folds: int, data_start: str) -> list[dict]:
    """
    Build expanding WF folds.
    Train always starts from data_start (or actual data start if later).
    Each fold test window is ~18 months.
    """
    actual_start = X.index[0].replace(tzinfo=None)
    config_start = pd.Timestamp(data_start)
    # Use later of config start vs actual data start
    effective_start = max(actual_start, config_start)

    data_end = X.index[-1].replace(tzinfo=None)  # naive for fold arithmetic
    total_months = (
        (data_end.year - effective_start.year) * 12 +
        data_end.month - effective_start.month
    )
    months_per_fold = max(total_months // (n_folds + 1), 6)
    data_start = effective_start.strftime("%Y-%m-%d")  # use effective start

    folds = []
    for i in range(n_folds):
        test_start_month = months_per_fold * (i + 1)
        test_end_month   = months_per_fold * (i + 2)

        train_end  = pd.Timestamp(data_start) + pd.DateOffset(months=test_start_month)
        test_start = train_end
        test_end   = pd.Timestamp(data_start) + pd.DateOffset(months=test_end_month)

        if test_end > data_end:
            test_end = data_end

        if test_start >= data_end:
            break

        folds.append({
            "fold":       i,
            "train_end":  train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end":   test_end.strftime("%Y-%m-%d"),
        })
    return folds


# ─── Regime gate ─────────────────────────────────────────────────────────────

def _regime_ok(row: pd.Series, cfg: dict) -> bool:
    """Check regime gate: weekly SMA-50 + ADX >= 20."""
    regime_col = cfg.get("regime_col", "1w_dist_sma_50")
    # ADX gate
    adx_col = "1d_adx"
    adx_ok = True
    if adx_col in row.index:
        adx_ok = float(row.get(adx_col, 0)) >= ADX_GATE

    # Regime: allow both bull and bear (bidirectional)
    # Only block when regime is undefined (NaN)
    regime_defined = pd.notna(row.get(regime_col, 0))

    return adx_ok and regime_defined


# ─── Backtest one fold ────────────────────────────────────────────────────────

def _backtest_fold(model, meta_model, X_test: pd.DataFrame,
                   y_test: pd.Series, cfg: dict,
                   threshold: float, le) -> pd.DataFrame:
    """
    Simulate trades on test fold.
    Returns DataFrame of individual trades with P&L.
    """
    tp_pct = cfg["tp_pct"] / 100
    sl_pct = cfg["sl_pct"] / 100
    time_limit = cfg["time_limit_days"]

    # Get 1D close prices for exit simulation
    from config import ohlcv_path
    symbol = cfg.get("symbol", "")
    df_1d_path = ohlcv_path(symbol, "1d")
    df_1d = pd.read_csv(df_1d_path, index_col=0, parse_dates=True).sort_index()
    df_1d.index = _norm_tz(df_1d.index)

    probas = model.predict_proba(X_test)
    # classes from LabelEncoder: [DOWN, SIDEWAYS, UP]
    classes = le.classes_.tolist()
    up_idx   = classes.index("UP")   if "UP"   in classes else 2
    down_idx = classes.index("DOWN") if "DOWN" in classes else 0

    trades = []
    trade_exit_date = None  # track when current trade closes

    for i, (ts, row) in enumerate(X_test.iterrows()):
        # Reset lock-out once trade has expired
        if trade_exit_date is not None and ts > trade_exit_date:
            trade_exit_date = None

        if trade_exit_date is not None:
            continue  # still in a trade

        if not _regime_ok(row, cfg):
            continue

        p_up   = probas[i, up_idx]
        p_down = probas[i, down_idx]

        # Meta-labeling gate (per-symbol threshold override via cfg["meta_win_thresh"])
        if meta_model is not None:
            meta_thresh = cfg.get("meta_win_thresh", META_WIN_THRESH)
            meta_proba = meta_model.predict_proba(row.values.reshape(1, -1))[0, 1]
            if meta_proba < meta_thresh:
                continue

        # Determine direction
        if p_up >= threshold:
            direction = "LONG"
            signal_strength = p_up
        elif p_down >= threshold:
            direction = "SHORT"
            signal_strength = p_down
        else:
            continue

        # Find entry price on 1D data
        entry_rows = df_1d[df_1d.index >= ts]
        if entry_rows.empty:
            continue
        entry_price = float(entry_rows.iloc[0]["close"])
        entry_date  = entry_rows.index[0]

        # Simulate exit
        future = df_1d[df_1d.index > entry_date].head(time_limit)
        pnl_pct = None
        exit_reason = "time"
        actual_exit_date = future.index[-1] if not future.empty else entry_date

        for _, frow in future.iterrows():
            high_pct = (float(frow["high"]) - entry_price) / entry_price
            low_pct  = (float(frow["low"])  - entry_price) / entry_price

            if direction == "LONG":
                if high_pct >= tp_pct:
                    pnl_pct = tp_pct; exit_reason = "TP"
                    actual_exit_date = frow.name; break
                if low_pct <= -sl_pct:
                    pnl_pct = -sl_pct; exit_reason = "SL"
                    actual_exit_date = frow.name; break
            else:  # SHORT
                if low_pct <= -tp_pct:
                    pnl_pct = tp_pct; exit_reason = "TP"
                    actual_exit_date = frow.name; break
                if high_pct >= sl_pct:
                    pnl_pct = -sl_pct; exit_reason = "SL"
                    actual_exit_date = frow.name; break

        if pnl_pct is None:
            exit_price = float(future.iloc[-1]["close"]) if not future.empty else entry_price
            if direction == "LONG":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

        trade_exit_date = actual_exit_date  # lock out until this date

        trades.append({
            "entry_date":    entry_date,
            "direction":     direction,
            "entry_price":   entry_price,
            "pnl_pct":       pnl_pct,
            "exit_reason":   exit_reason,
            "signal_strength": signal_strength,
        })

    return pd.DataFrame(trades)


# ─── Fold metrics ─────────────────────────────────────────────────────────────

def _compute_metrics(trades: pd.DataFrame) -> dict:
    if len(trades) < MIN_TRADES:
        return {"sharpe": -99, "wr": 0, "n_trades": len(trades), "pf": 0}

    rets = trades["pnl_pct"].values
    wr   = (rets > 0).mean()
    wins = rets[rets > 0].sum()
    loss = abs(rets[rets < 0].sum())
    pf   = wins / (loss + 1e-9)
    sharpe = rets.mean() / (rets.std() + 1e-9) * np.sqrt(252 / 5)  # annualised

    long_trades  = (trades["direction"] == "LONG").sum()
    short_trades = (trades["direction"] == "SHORT").sum()

    return {
        "sharpe":       round(sharpe, 3),
        "wr":           round(wr, 3),
        "n_trades":     len(trades),
        "profit_factor": round(pf, 3),
        "long_trades":  int(long_trades),
        "short_trades": int(short_trades),
        "avg_pnl":      round(float(rets.mean()), 4),
    }


def _viability(fold_metrics: list[dict]) -> str:
    # Exclude folds below MIN_TRADES (-99 sentinel) from averages
    valid = [m for m in fold_metrics if m["n_trades"] >= MIN_TRADES]
    if not valid:
        return "NOT_VIABLE"

    # Require at least 2 valid folds — a single fold can't establish consistency
    if len(valid) < 2:
        return "NOT_VIABLE"

    positive   = sum(1 for m in valid if m["sharpe"] > 0)
    avg_sharpe = np.mean([m["sharpe"] for m in valid])
    avg_wr     = np.mean([m["wr"] for m in valid])
    n_folds    = len(valid)

    if (avg_sharpe >= VIABLE_SHARPE and avg_wr >= VIABLE_WR
            and positive >= max(int(n_folds * 0.6), 2)):
        return "VIABLE"
    if (avg_sharpe >= MARGINAL_SHARPE and avg_wr >= MARGINAL_WR
            and positive >= max(int(n_folds * 0.5), 2)):
        return "MARGINAL"
    # High-Sharpe secondary path: excellent risk-adjusted returns at lower hit-rate.
    # At 3:1 R:R, 33% WR = ~0.32% EV/trade (still positive). Sharpe >= 0.70 guards
    # against single-lucky-year systems — requires multi-fold consistency.
    if (avg_sharpe >= MARGINAL_SHARPE_HIGH and avg_wr >= MARGINAL_WR_LOW
            and positive >= max(int(n_folds * 0.5), 2)):
        return "MARGINAL"
    return "NOT_VIABLE"


# ─── Meta model ──────────────────────────────────────────────────────────────

def _train_meta(X_train: pd.DataFrame, y_train: pd.Series,
                trades: pd.DataFrame) -> object | None:
    """
    Train meta-labeling model: predicts P(win) for each signal.
    Uses last META_TRAIN_FRAC of training window.
    """
    import lightgbm as lgb

    if trades.empty:
        return None

    # Build meta labels: 1 if trade won, 0 if lost
    meta_X = X_train.loc[X_train.index.isin(
        pd.DatetimeIndex(trades["entry_date"].values)
    )]
    if len(meta_X) < 20:
        return None

    meta_y = (trades.set_index("entry_date")["pnl_pct"] > 0).astype(int)
    meta_y = meta_y[meta_y.index.isin(meta_X.index)]
    meta_X = meta_X.loc[meta_X.index.isin(meta_y.index)]

    if len(meta_X) < 20:
        return None

    model = lgb.LGBMClassifier(**META_LGBM_PARAMS, objective="binary")
    model.fit(meta_X, meta_y)
    return model


# ─── Main WF loop ─────────────────────────────────────────────────────────────

def run_wf(symbol: str) -> dict:
    """Run full walk-forward validation for one symbol."""
    import lightgbm as lgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import LabelEncoder

    cfg = INSTRUMENTS.get(symbol)
    if cfg is None:
        log.error(f"Unknown symbol: {symbol}")
        return {}

    cfg = {**cfg, "symbol": symbol}

    result = _load_features_and_labels(symbol)
    if result is None:
        return {}

    X, y, le = result
    folds = _build_folds(X, cfg["wf_folds"], cfg["data_start"])
    log.info(f"{symbol}: {len(folds)} folds")

    best_threshold = None
    best_sharpe    = -99
    fold_results   = []
    last_model     = None

    for threshold in WF_THRESHOLD_GRID:
        fold_metrics = []

        for fold in folds:
            train_mask = X.index <= _ts(fold["train_end"])
            test_mask  = (X.index >= _ts(fold["test_start"])) & (X.index <= _ts(fold["test_end"]))

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_te, y_te = X[test_mask],  y[test_mask]

            if len(X_tr) < 200 or len(X_te) < 20:
                continue

            # Train primary model (uses lighter params in --fast mode)
            lgbm = lgb.LGBMClassifier(**_WF_PARAMS,
                                       objective="multiclass", num_class=3)
            model = CalibratedClassifierCV(lgbm, method="isotonic", cv=_WF_CV)
            model.fit(X_tr, y_tr)

            # Meta model
            meta_model = None
            if META_LABELING:
                # Use last META_TRAIN_FRAC of training window for meta
                meta_start = X_tr.index[int(len(X_tr) * (1 - META_TRAIN_FRAC))]
                X_meta = X_tr[X_tr.index >= meta_start]
                y_meta = y_tr[y_tr.index >= meta_start]
                # Run a quick backtest on meta window to get trade labels
                fold_trades = _backtest_fold(
                    model, None, X_meta, y_meta, cfg, threshold, le
                )
                meta_model = _train_meta(X_meta, y_meta, fold_trades)

            trades = _backtest_fold(model, meta_model, X_te, y_te,
                                     cfg, threshold, le)
            metrics = _compute_metrics(trades)
            metrics["fold"] = fold["fold"]
            fold_metrics.append(metrics)

            # Keep model from last fold
            last_model = model

            log.info(
                f"  {symbol} fold={fold['fold']} thresh={threshold:.2f} "
                f"Sharpe={metrics['sharpe']:.3f} WR={metrics['wr']:.1%} "
                f"Trades={metrics['n_trades']}"
            )

        if not fold_metrics:
            continue

        # Use only valid folds (>= MIN_TRADES) for threshold selection.
        # Require ≥2 valid folds so the threshold selector and _viability() agree —
        # a single-fold threshold will always be NOT_VIABLE and should not be preferred.
        valid_metrics = [m for m in fold_metrics if m["n_trades"] >= MIN_TRADES]
        if len(valid_metrics) < 2:
            continue
        avg_sharpe = np.mean([m["sharpe"] for m in valid_metrics])
        if avg_sharpe > best_sharpe:
            best_sharpe    = avg_sharpe
            best_threshold = threshold
            fold_results   = fold_metrics

    # Determine viability (valid folds only)
    valid_results = [m for m in fold_results if m["n_trades"] >= MIN_TRADES] if fold_results else []
    verdict = _viability(fold_results) if fold_results else "NOT_VIABLE"
    avg_wr  = np.mean([m["wr"] for m in valid_results]) if valid_results else 0
    total_trades = sum(m["n_trades"] for m in fold_results)

    log.info(
        f"\n{'='*60}\n{symbol} RESULT: {verdict}\n"
        f"Best threshold: {best_threshold}\n"
        f"Avg Sharpe: {best_sharpe:.3f}  Avg WR: {avg_wr:.1%}\n"
        f"Total trades: {total_trades}\n{'='*60}"
    )

    # Save best model (retrain on all data with best threshold)
    if last_model and verdict in ("VIABLE", "MARGINAL"):
        lgbm = lgb.LGBMClassifier(**LGBM_PARAMS, objective="multiclass", num_class=3)
        final_model = CalibratedClassifierCV(lgbm, method="isotonic", cv=3)
        final_model.fit(X, y)

        os.makedirs(os.path.join(MODELS_DIR, symbol), exist_ok=True)
        joblib.dump(final_model, model_path(symbol, "wf_decision_model_v2.pkl"))
        with open(features_list_path(symbol), "w") as f:
            f.write("\n".join(X.columns.tolist()))
        log.info(f"{symbol}: WF model saved")

    # Save results JSON
    wf_result = {
        "symbol":          symbol,
        "verdict":         verdict,
        "best_threshold":  best_threshold,
        "avg_sharpe":      round(best_sharpe, 3),
        "avg_wr":          round(avg_wr, 3),
        "total_trades":    total_trades,
        "fold_results":    fold_results,
        "n_folds":         len(fold_results),
    }
    result_path = os.path.join(MODELS_DIR, symbol, "wf_results.json")
    os.makedirs(os.path.join(MODELS_DIR, symbol), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(wf_result, f, indent=2, default=str)

    return wf_result


# ─── Portfolio summary ────────────────────────────────────────────────────────

def portfolio_summary(results: dict) -> None:
    print("\n" + "="*70)
    print(f"{'SYMBOL':<14} {'VERDICT':<14} {'SHARPE':>8} {'WR':>7} {'TRADES':>7} {'THRESH':>7}")
    print("-"*70)
    for sym, r in sorted(results.items(), key=lambda x: x[1].get("avg_sharpe", -99), reverse=True):
        print(
            f"{sym:<14} {r.get('verdict','?'):<14} "
            f"{r.get('avg_sharpe',0):>8.3f} "
            f"{r.get('avg_wr',0):>7.1%} "
            f"{r.get('total_trades',0):>7} "
            f"{r.get('best_threshold',0):>7.2f}"
        )
    print("="*70)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    global _WF_PARAMS, _WF_CV

    parser = argparse.ArgumentParser(description="Walk-forward validation for India stocks")
    parser.add_argument("--symbol",  default=None, help="Single symbol")
    parser.add_argument("--fast",    action="store_true",
                        help="Use lighter LightGBM params (3-5x faster, slightly less precise)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel symbols (default 4, use 1 for sequential)")
    args = parser.parse_args()

    if args.fast:
        _WF_PARAMS = WF_FAST_LGBM_PARAMS
        _WF_CV     = 2
        log.info("Fast mode: n_estimators=200, cv=2")

    symbols = [args.symbol] if args.symbol else ALL_SYMBOLS
    all_results = {}

    if args.symbol or args.workers == 1:
        # Single symbol or explicit sequential
        for sym in symbols:
            result = run_wf(sym)
            if result:
                all_results[sym] = result
    else:
        # Parallel symbol processing
        log.info(f"Running {len(symbols)} symbols with {args.workers} parallel workers")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_wf, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    result = fut.result()
                    if result:
                        all_results[sym] = result
                except Exception as e:
                    log.error(f"{sym} WF failed: {e}")

    if len(all_results) > 1:
        portfolio_summary(all_results)

    log.info("Walk-forward validation complete.")


if __name__ == "__main__":
    main()
