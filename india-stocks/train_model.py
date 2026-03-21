"""
LightGBM Training — India Stocks
3-class labels: UP / DOWN / SIDEWAYS (same as crypto system).
LightGBM + isotonic calibration.

Usage:
    python train_model.py                   # all symbols
    python train_model.py --symbol NIFTY50  # single
    python train_model.py --symbol NIFTY50 --wf  # walk-forward only
"""

import os
import sys
import argparse
import logging
import joblib

import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INSTRUMENTS, ALL_SYMBOLS, MODELS_DIR,
    features_path, model_path, features_list_path,
    LGBM_PARAMS, META_LGBM_PARAMS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Label generation ─────────────────────────────────────────────────────────

def make_labels(df_1d: pd.DataFrame, symbol: str) -> pd.Series:
    """
    3-class forward return labels aligned to 1H feature timestamps.
    Uses 1D close prices to compute future return.
    Labels: UP=2, SIDEWAYS=1, DOWN=0
    """
    cfg = INSTRUMENTS[symbol]
    threshold = cfg["direction_threshold"] / 100.0
    time_limit = cfg["time_limit_days"]

    closes = df_1d["close"].copy()
    future_ret = closes.shift(-time_limit) / closes - 1

    labels = pd.Series("SIDEWAYS", index=closes.index)
    labels[future_ret >  threshold] = "UP"
    labels[future_ret < -threshold] = "DOWN"

    return labels


def _load_features_and_labels(symbol: str) -> tuple[pd.DataFrame, pd.Series] | None:
    """Load feature CSV and compute labels from 1D price data."""
    feat_path = features_path(symbol)
    if not os.path.exists(feat_path):
        log.error(f"{symbol}: features file not found")
        return None

    # Load 1D data for labels
    import pytz
    from config import ohlcv_path
    IST = pytz.timezone("Asia/Kolkata")

    def _norm(idx):
        if idx.tz is None:
            return idx.tz_localize(IST)
        return idx.tz_convert(IST)

    df_1d_path = ohlcv_path(symbol, "1d")
    if not os.path.exists(df_1d_path):
        log.error(f"{symbol}: 1D OHLCV not found for labels")
        return None

    df_feat = pd.read_csv(feat_path, index_col=0, parse_dates=True).sort_index()
    df_1d   = pd.read_csv(df_1d_path, index_col=0, parse_dates=True).sort_index()

    df_feat.index = _norm(df_feat.index)
    df_1d.index   = _norm(df_1d.index)

    # Build labels on daily index, merge to features via asof
    labels_daily = make_labels(df_1d, symbol)
    labels_daily.name = "label"
    labels_daily.index = _norm(labels_daily.index)
    labels_daily.index.name = "timestamp"

    aligned = pd.merge_asof(
        df_feat.reset_index(),
        labels_daily.reset_index(),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    aligned = aligned.dropna(subset=["label"])
    y = aligned["label"]
    X = aligned.drop(columns=["label"])

    # Drop columns with >30% NaN
    X = X.loc[:, X.isna().mean() < 0.3]
    X = X.ffill().fillna(0)

    # Encode labels
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), index=y.index, name="label")
    # classes: 0=DOWN, 1=SIDEWAYS, 2=UP (alphabetical order from LabelEncoder)

    return X, y_enc, le


# ─── Train standard model ────────────────────────────────────────────────────

def train_model(symbol: str, save: bool = True) -> dict:
    """Train LightGBM + isotonic calibration on full available data."""
    import lightgbm as lgb

    result = _load_features_and_labels(symbol)
    if result is None:
        return {}

    X, y, le = result
    log.info(f"{symbol}: training on {len(X)} rows × {X.shape[1]} features")
    log.info(f"{symbol}: label dist = {y.value_counts().to_dict()}")

    lgbm = lgb.LGBMClassifier(**LGBM_PARAMS, objective="multiclass", num_class=3)
    model = CalibratedClassifierCV(lgbm, method="isotonic", cv=3)
    model.fit(X, y)

    # Feature importance
    feat_importance = pd.Series(
        lgbm.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    log.info(f"{symbol}: top 10 features:\n{feat_importance.head(10)}")

    if save:
        os.makedirs(os.path.join(MODELS_DIR, symbol), exist_ok=True)
        joblib.dump(model, model_path(symbol, "decision_model_v2.pkl"))
        with open(features_list_path(symbol), "w") as f:
            f.write("\n".join(X.columns.tolist()))
        log.info(f"{symbol}: model saved → {model_path(symbol, 'decision_model_v2.pkl')}")

    return {"symbol": symbol, "n_samples": len(X), "features": X.shape[1],
            "classes": le.classes_.tolist()}


# ─── Walk-forward trained model (latest fold only) ───────────────────────────

def train_wf_model(symbol: str, train_end: str,
                   X: pd.DataFrame, y: pd.Series,
                   save_suffix: str = "wf_decision_model_v2") -> object | None:
    """
    Train model on data up to train_end date.
    Called by walk_forward.py for each fold.
    Returns calibrated model.
    """
    import lightgbm as lgb

    mask = X.index <= train_end
    X_tr, y_tr = X[mask], y[mask]

    if len(X_tr) < 200:
        log.warning(f"{symbol}: insufficient training data ({len(X_tr)} rows)")
        return None

    lgbm = lgb.LGBMClassifier(**LGBM_PARAMS, objective="multiclass", num_class=3)
    model = CalibratedClassifierCV(lgbm, method="isotonic", cv=3)
    model.fit(X_tr, y_tr)
    return model


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train India stocks LightGBM model")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--wf",    action="store_true",
                        help="Run walk-forward validation instead")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else ALL_SYMBOLS

    if args.wf:
        from walk_forward import run_wf
        for sym in symbols:
            run_wf(sym)
    else:
        for sym in symbols:
            train_model(sym)

    log.info("Training complete.")


if __name__ == "__main__":
    main()
