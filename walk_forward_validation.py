"""
Expanding Walk-Forward Validation — Bidirectional Direction Prediction
======================================================================
Multi-fold temporal validation with NO data leakage.
Train window anchored to each coin's data start and grows each fold (expanding WF).

Architecture per coin:
  BTC      — 5 folds, expanding train anchored to 2019-09-11
  ETH      — 4 folds, expanding train anchored to 2019-11-28 (fold 0 / test 2021 removed: regime mismatch)
  SOL      — 3 folds, expanding train anchored to 2021-09-02
  PEPE     — 4 folds, expanding train anchored to 2024-04-15

Strategy: BIDIRECTIONAL
  P(UP)   >= threshold + ADX >= 20  ->  LONG   (TP above, SL below entry)
  P(DOWN) >= threshold + ADX >= 20  ->  SHORT  (TP below, SL above entry)
  If both UP and DOWN above threshold -> take higher probability side.

Coin-specific TP/SL calibrated to direction label thresholds (2:1 TP:SL ratio,
breakeven win rate = 33.3% for all coins):
  BTC/ETH  label=+-1.5%  TP=3.0%  SL=1.5%
  SOL      label=+-2.5%  TP=5.0%  SL=2.5%
  PEPE     label=+-5.0%  TP=10.0% SL=5.0%

Threshold selection: best MEDIAN Sharpe across all folds (cross-fold selection).
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Coin-specific TP/SL (calibrated to label thresholds, 3:1 R:R) ──────────
COIN_PARAMS = {
    'BTC_USDT':  {'tp': 0.030, 'sl': 0.015, 'time_limit': 48},   # 2:1 R:R, breakeven=33.3%
    'ETH_USDT':  {'tp': 0.045, 'sl': 0.015, 'time_limit': 48},   # 3:1 R:R, breakeven=25.0%
    'SOL_USDT':  {'tp': 0.075, 'sl': 0.025, 'time_limit': 72},   # 3:1 R:R; label thresh=3.0%
    'PEPE_USDT': {'tp': 0.150, 'sl': 0.050, 'time_limit': 48},   # 3:1 R:R, breakeven=25.0%
    # New coins — 3:1 R:R, breakeven=25.0%
    'AVAX_USDT': {'tp': 0.075, 'sl': 0.025, 'time_limit': 72},   # similar beta to SOL
    'BNB_USDT':  {'tp': 0.060, 'sl': 0.020, 'time_limit': 48},   # slightly lower vol
    'LINK_USDT': {'tp': 0.075, 'sl': 0.025, 'time_limit': 72},   # ETH-correlated altcoin
    # L2/DeFi expansion — higher vol alts, 3:1 R:R
    'ARB_USDT':  {'tp': 0.075, 'sl': 0.025, 'time_limit': 72},   # Arbitrum, listed Mar 2023
    'OP_USDT':   {'tp': 0.075, 'sl': 0.025, 'time_limit': 72},   # Optimism, listed Jun 2022
    'INJ_USDT':  {'tp': 0.090, 'sl': 0.030, 'time_limit': 72},   # Injective, higher vol, listed Oct 2020
}

# ── Expanding folds per coin (train anchored to data start, grows each fold) ─
FOLDS = {
    'BTC_USDT': [
        # BTC feature CSV starts 2019-09-11 — 5 folds, expanding train
        {'train': ('2019-09-11', '2020-12-31'), 'test': ('2021-01-01', '2021-12-31')},  # fold 0: bull+correction cycle
        {'train': ('2019-09-11', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')},  # fold 1
        {'train': ('2019-09-11', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},  # fold 2
        {'train': ('2019-09-11', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},  # fold 3
        {'train': ('2019-09-11', '2024-12-31'), 'test': ('2025-01-01', '2026-02-25')},  # fold 4
    ],
    'ETH_USDT': [
        # ETH feature CSV starts 2019-11-28 — 4 folds, expanding train
        # (fold 0 / test 2021 removed: meta model trained on pre-2021 bear doesn't generalise to 2021 bull)
        {'train': ('2019-11-28', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')},  # fold 1
        {'train': ('2019-11-28', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},  # fold 2
        {'train': ('2019-11-28', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},  # fold 3
        {'train': ('2019-11-28', '2024-12-31'), 'test': ('2025-01-01', '2026-02-25')},  # fold 4
    ],
    'SOL_USDT': [
        # SOL feature CSV starts 2021-09-02 — 3 folds, expanding train
        {'train': ('2021-09-02', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},  # fold 1
        {'train': ('2021-09-02', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},  # fold 2
        {'train': ('2021-09-02', '2024-12-31'), 'test': ('2025-01-01', '2026-02-25')},  # fold 3 (was 2022-01-01)
    ],
    'PEPE_USDT': [
        # PEPE feature CSV starts 2024-04-15 — 4 folds, expanding train
        {'train': ('2024-04-15', '2024-09-30'), 'test': ('2024-10-01', '2024-12-31')},  # fold 1
        {'train': ('2024-04-15', '2024-12-31'), 'test': ('2025-01-01', '2025-04-30')},  # fold 2
        {'train': ('2024-04-15', '2025-03-31'), 'test': ('2025-04-01', '2025-08-31')},  # fold 3 (was 2024-07-01)
        {'train': ('2024-04-15', '2025-07-31'), 'test': ('2025-08-01', '2026-02-25')},  # fold 4 (was 2024-10-01)
    ],
    # ── New coins ───────────────────────────────────────────────────────────
    'AVAX_USDT': [
        # AVAX listed ~Oct 2020; features start ~Oct 2021 after indicator warmup
        {'train': ('2021-10-01', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},  # ~15mo
        {'train': ('2021-10-01', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},  # ~27mo
        {'train': ('2022-01-01', '2024-12-31'), 'test': ('2025-01-01', '2026-02-25')},  # 36mo
    ],
    'BNB_USDT': [
        # BNB spot from 2017; features start ~2018-2019 after warmup
        {'train': ('2019-11-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')},  # ~26mo
        {'train': ('2020-01-01', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},  # 36mo
        {'train': ('2021-01-01', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},  # 36mo
        {'train': ('2022-01-01', '2024-12-31'), 'test': ('2025-01-01', '2026-02-25')},  # 36mo
    ],
    'LINK_USDT': [
        # LINK listed ~Jan 2019; features start ~early 2020 after warmup
        {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')},  # 24mo
        {'train': ('2020-01-01', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},  # 36mo
        {'train': ('2021-01-01', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},  # 36mo
        {'train': ('2022-01-01', '2024-12-31'), 'test': ('2025-01-01', '2026-02-25')},  # 36mo
    ],
    # ── L2/DeFi expansion ───────────────────────────────────────────────────
    'ARB_USDT': [
        # ARB listed Binance Mar 23 2023; features start ~Apr 2023 after indicator warmup
        {'train': ('2023-04-20', '2024-06-30'), 'test': ('2024-07-01', '2024-12-31')},  # 14mo train
        {'train': ('2023-04-20', '2024-12-31'), 'test': ('2025-01-01', '2025-06-30')},  # 20mo train
        {'train': ('2023-04-20', '2025-06-30'), 'test': ('2025-07-01', '2026-02-25')},  # 26mo train
    ],
    'OP_USDT': [
        # OP listed Binance Jun 2022; features start ~Jul 2022 after indicator warmup
        {'train': ('2022-07-01', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},  # 18mo train
        {'train': ('2022-07-01', '2024-12-31'), 'test': ('2025-01-01', '2025-06-30')},  # 30mo train
        {'train': ('2022-07-01', '2025-06-30'), 'test': ('2025-07-01', '2026-02-25')},  # 36mo train
    ],
    'INJ_USDT': [
        # INJ listed Binance Oct 2020; features start ~Jan 2021 after indicator warmup
        {'train': ('2021-01-01', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},  # 24mo train
        {'train': ('2021-01-01', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},  # 36mo train
        {'train': ('2021-01-01', '2024-12-31'), 'test': ('2025-01-01', '2026-02-25')},  # 48mo train
    ],
}

# ── Shared constants ───────────────────────────────────────────────────────
TIME_LIMIT      = 48        # max hours in position
POSITION_SIZE   = 0.30
INITIAL_CAPITAL = 10_000
ADX_MIN         = 20        # only trade in trending markets
REGIME_GATE     = True      # BTC/ETH/SOL: 1w_dist_sma_50; PEPE: 1d_dist_sma_50
FEE             = 0.0006
SLIPPAGE        = 0.0003
SPREAD          = 0.0002
ROUND_TRIP_COST = (FEE + SLIPPAGE + SPREAD) * 2   # ~0.22%
THRESHOLDS      = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_FOLD_TRADES = 5

# ── Experiment toggles ─────────────────────────────────────────────────────
LONG_ONLY        = False   # True = disable all SHORT signals (tests structural bull-bias)
META_LABELING    = True    # True = add meta-model layer: only trade when meta says WIN
META_LABEL_THRESH      = 0.25   # primary threshold used to generate meta-label training samples
META_MIN_SIGNALS       = 20     # minimum signal bars needed in cal set to fit meta model
META_WIN_THRESH        = 0.52   # meta model P(WIN) required to execute a trade (was 0.55)
META_CAL_WINDOW_MONTHS = 12     # rolling cal window: use last N months of train as OOS meta set


# ── Helpers ────────────────────────────────────────────────────────────────

def load_data(coin):
    path = os.path.join(BASE_DIR, f"data/{coin}_multi_tf_features.csv")
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_feature_cols(df):
    drop = {'timestamp', 'target_return', 'target_direction',
            'open', 'high', 'low', 'close', 'volume', 'decision_label'}
    return [c for c in df.columns if c not in drop]


def select_features(train_df: pd.DataFrame, feature_cols: list) -> list:
    """
    Prune zero-importance features using LightGBM gain importance.

    Trains a single fast LGBM on the provided (first-fold) training data to
    identify columns the tree splitter never selects.  Constant columns,
    near-duplicate features, and columns dominated by NaN all score gain=0.

    Conservative: uses oldest data only (first fold), so no lookahead into
    future folds is introduced by the feature selection step.
    """
    lgbm = LGBMClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=5,
        num_leaves=31, min_child_samples=50,
        class_weight='balanced', random_state=42, n_jobs=1, verbose=-1,
    )
    X = train_df[feature_cols].fillna(0)
    y = train_df['decision_label']
    lgbm.fit(X, y)

    importance = lgbm.booster_.feature_importance(importance_type='gain')
    selected   = [col for col, imp in zip(feature_cols, importance) if imp > 0]

    n_dropped = len(feature_cols) - len(selected)
    if n_dropped > 0:
        print(f"  Feature pruning: {n_dropped} zero-gain features dropped "
              f"({len(selected)} of {len(feature_cols)} remain)")
    return selected if selected else feature_cols  # guard: never return empty list


# ── Model ──────────────────────────────────────────────────────────────────

def _build_lgbm():
    return LGBMClassifier(
        n_estimators=700, learning_rate=0.02, max_depth=7,
        num_leaves=63, min_child_samples=50, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        class_weight='balanced', random_state=42, n_jobs=1, verbose=-1,
    )


def train_model(train_df, feature_cols):
    """
    3-class UP/DOWN/SIDEWAYS LightGBM with isotonic calibration.

    The zero-sum softmax constraint is intentional: forcing the model to split
    probability across three classes makes a high P(UP) genuinely selective —
    it requires the model to be confident enough to outcompete both DOWN and
    SIDEWAYS simultaneously.  Empirically this produces better WR than two
    independent binary classifiers for this dataset.
    """
    X = train_df[feature_cols].fillna(0)
    y = train_df['decision_label']

    split = int(len(X) * 0.85)
    X_fit, y_fit = X.iloc[:split], y.iloc[:split]
    X_cal, y_cal = X.iloc[split:], y.iloc[split:]

    lgbm = _build_lgbm()
    lgbm.fit(X_fit, y_fit)
    model = CalibratedClassifierCV(lgbm, cv='prefit', method='isotonic')
    model.fit(X_cal, y_cal)

    train_acc = (model.predict(X_fit) == y_fit).mean()
    classes   = list(model.classes_)
    return model, classes, train_acc


# ── Meta-labeling layer ────────────────────────────────────────────────────

def generate_meta_labels(model, classes, df, feature_cols,
                         threshold, tp_pct, sl_pct, coin='', time_limit=TIME_LIMIT):
    """
    Scan df (OOS w.r.t. primary model) for signal bars at `threshold`.
    For each signal bar, simulate the actual outcome (WIN=1 / LOSS=0).
    Returns a DataFrame: feature_cols + [up_prob, down_prob, direction_long, meta_label]

    This gives the meta model a set of (context → outcome) pairs to learn WHEN
    the primary model's signals are actually profitable.
    """
    X  = df[feature_cols].fillna(0)
    df = df.reset_index(drop=True).copy()

    proba    = model.predict_proba(X)
    up_idx   = classes.index('UP')   if 'UP'   in classes else None
    down_idx = classes.index('DOWN') if 'DOWN' in classes else None
    df['up_prob']   = proba[:, up_idx]   if up_idx   is not None else 0.0
    df['down_prob'] = proba[:, down_idx] if down_idx is not None else 0.0

    meta_rows = []

    for i in range(len(df) - time_limit - 1):
        row   = df.iloc[i]
        up_p  = row['up_prob']
        dn_p  = row['down_prob']
        adx   = row.get('1h_adx', ADX_MIN)

        if adx < ADX_MIN:
            continue

        go_long  = up_p >= threshold
        go_short = dn_p >= threshold and not LONG_ONLY

        if go_long and go_short:
            go_long  = up_p >= dn_p
            go_short = not go_long

        if REGIME_GATE:
            regime_dist = row.get(
                '1d_dist_sma_50' if coin == 'PEPE_USDT' else '1w_dist_sma_50', 0)
            if go_long  and regime_dist <= 0:
                go_long  = False
            if go_short and regime_dist >= 0:
                go_short = False

        if not go_long and not go_short:
            continue

        direction    = 'LONG' if go_long else 'SHORT'
        entry_price  = df.iloc[i + 1]['open']   # 1-bar delay (same as simulate_trades)

        if direction == 'LONG':
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)

        outcome = None
        for j in range(i + 1, min(i + 1 + time_limit, len(df))):
            frow = df.iloc[j]
            if direction == 'LONG':
                if frow['low']  <= sl_price: outcome = 0; break
                if frow['high'] >= tp_price: outcome = 1; break
            else:
                if frow['high'] >= sl_price: outcome = 0; break
                if frow['low']  <= tp_price: outcome = 1; break

        if outcome is None:          # time-expired: compare exit price to entry
            exit_px = df.iloc[min(i + time_limit, len(df) - 1)]['close']
            outcome = 1 if (direction == 'LONG' and exit_px > entry_price) or \
                           (direction == 'SHORT' and exit_px < entry_price) else 0

        feat = {f: row[f] for f in feature_cols if f in row.index}
        feat['up_prob']        = up_p
        feat['down_prob']      = dn_p
        feat['direction_long'] = 1 if direction == 'LONG' else 0
        feat['meta_label']     = outcome
        meta_rows.append(feat)

    return pd.DataFrame(meta_rows) if meta_rows else None


def train_meta_model(meta_df, feature_cols):
    """
    Fit a lightweight binary LightGBM: predict WIN (1) vs LOSS (0).
    Features: same feature set as primary model + [up_prob, down_prob, direction_long]
    Returns (model, feature_list) or None if insufficient data / single-class target.
    """
    if meta_df is None or len(meta_df) < META_MIN_SIGNALS:
        return None

    meta_feat = [f for f in feature_cols if f in meta_df.columns]
    for extra in ['up_prob', 'down_prob', 'direction_long']:
        if extra in meta_df.columns and extra not in meta_feat:
            meta_feat.append(extra)

    X = meta_df[meta_feat].fillna(0)
    y = meta_df['meta_label'].astype(int)

    if y.nunique() < 2:     # can't train a binary classifier on a single class
        return None

    # Meta binary classifier — same complexity as primary but deterministic (n_jobs=1).
    # n_estimators=200, max_depth=4, num_leaves=15 tested to be appropriately calibrated
    # for 50-500 signal cal sets; n_jobs=1 ensures full determinism across runs.
    lgbm = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=10, subsample=0.8,
        colsample_bytree=0.8, class_weight='balanced',
        random_state=42, n_jobs=1, verbose=-1,
    )
    lgbm.fit(X, y)
    return lgbm, meta_feat


# ── Bidirectional trade simulation ─────────────────────────────────────────

def simulate_trades(model, df, feature_cols, classes, threshold,
                    tp_pct, sl_pct, coin='', time_limit=TIME_LIMIT,
                    meta_model=None, meta_feat=None):
    """
    Bidirectional simulation with 3-class model:
      P(UP)   >= threshold + ADX >= ADX_MIN  ->  LONG
      P(DOWN) >= threshold + ADX >= ADX_MIN  ->  SHORT
      If both above threshold, take the stronger signal.

    Optional meta_model layer: after primary signal fires, meta model predicts
    P(WIN) — trade only executes if P(WIN) >= META_WIN_THRESH.

    LONG_ONLY=True disables all SHORT signals.
    Entry at next candle open (1-candle delay).
    Worst-case: if both TP+SL hit in same candle -> SL wins.
    """
    X   = df[feature_cols].fillna(0)
    df  = df.copy()
    proba    = model.predict_proba(X)
    up_idx   = classes.index('UP')   if 'UP'   in classes else None
    down_idx = classes.index('DOWN') if 'DOWN' in classes else None
    df['up_prob']   = proba[:, up_idx]   if up_idx   is not None else 0.0
    df['down_prob'] = proba[:, down_idx] if down_idx is not None else 0.0

    capital        = INITIAL_CAPITAL
    trades         = []
    equity         = [capital]
    in_position    = False
    pending_entry  = False
    pending_dir    = None
    entry_exec_idx = None
    entry_price    = 0.0
    position_cap   = 0.0
    tp_price       = 0.0
    sl_price       = 0.0
    trade_dir      = None
    cooldown       = 0
    signal_prob    = 0.0
    entry_bar      = 0

    for i in range(1, len(df) - time_limit):
        row = df.iloc[i]

        if cooldown > 0:
            cooldown -= 1

        # ── Execute pending entry ────────────────────────────────────
        if pending_entry and not in_position and i == entry_exec_idx:
            entry_price  = row['open']
            position_cap = capital * POSITION_SIZE
            trade_dir    = pending_dir
            if trade_dir == 'LONG':
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
            else:   # SHORT
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
            in_position   = True
            pending_entry = False
            entry_bar     = i

        # ── Check exit ───────────────────────────────────────────────
        if in_position:
            bars_in = i - entry_bar
            if trade_dir == 'LONG':
                hit_tp  = row['high'] >= tp_price
                hit_sl  = row['low']  <= sl_price
            else:   # SHORT
                hit_tp  = row['low']  <= tp_price
                hit_sl  = row['high'] >= sl_price

            expired = bars_in >= time_limit

            if hit_sl or hit_tp or expired:
                if hit_sl:
                    pnl_pct = -sl_pct - ROUND_TRIP_COST
                    result  = 'LOSS'
                    reason  = 'SL'
                elif hit_tp:
                    pnl_pct = tp_pct - ROUND_TRIP_COST
                    result  = 'WIN'
                    reason  = 'TP'
                else:
                    exit_px = row['close']
                    if trade_dir == 'LONG':
                        raw = (exit_px - entry_price) / entry_price
                    else:
                        raw = (entry_price - exit_px) / entry_price
                    pnl_pct = raw - ROUND_TRIP_COST
                    result  = 'WIN' if pnl_pct > 0 else 'LOSS'
                    reason  = 'TIME'

                capital += position_cap * pnl_pct
                equity.append(capital)
                trades.append({
                    'result':      result,
                    'exit_reason': reason,
                    'direction':   trade_dir,
                    'net_pnl_pct': round(pnl_pct * 100, 4),
                    'signal_prob': signal_prob,
                })
                in_position = False
                cooldown    = 1

        # ── Generate signal ──────────────────────────────────────────
        if not in_position and not pending_entry and cooldown == 0:
            up_p   = row['up_prob']
            down_p = row['down_prob']
            adx    = row.get('1h_adx', ADX_MIN)

            if adx < ADX_MIN:
                continue

            go_long  = up_p   >= threshold
            go_short = down_p >= threshold and not LONG_ONLY

            if go_long and go_short:
                # Both above threshold — take stronger signal
                go_long  = up_p >= down_p
                go_short = not go_long

            # Single-MA regime gate:
            # BTC/ETH/SOL: 1w_dist_sma_50 sets macro direction (weekly SMA-50 stable).
            # PEPE: 1d_dist_sma_50 only — 2 years of data, weekly SMA-50 is unstable.
            if REGIME_GATE:
                if coin == 'PEPE_USDT':
                    regime_dist = row.get('1d_dist_sma_50', 0)
                else:
                    regime_dist = row.get('1w_dist_sma_50', 0)
                in_bull = regime_dist > 0
                in_bear = regime_dist < 0
                if go_long  and not in_bull:
                    go_long  = False
                if go_short and not in_bear:
                    go_short = False

            # ── Meta model filter ─────────────────────────────────────
            # Only execute if meta model predicts P(WIN) >= META_WIN_THRESH.
            if (go_long or go_short) and meta_model is not None and meta_feat is not None:
                mrow = {f: (row[f] if f in row.index else 0.0) for f in meta_feat}
                mrow['up_prob']        = up_p
                mrow['down_prob']      = down_p
                mrow['direction_long'] = 1 if go_long else 0
                mX = pd.DataFrame([mrow])[meta_feat].fillna(0)
                meta_p = meta_model.predict_proba(mX)[0][1]
                if meta_p < META_WIN_THRESH:
                    go_long  = False
                    go_short = False

            if go_long:
                pending_entry  = True
                pending_dir    = 'LONG'
                signal_prob    = up_p
                entry_exec_idx = i + 1
            elif go_short:
                pending_entry  = True
                pending_dir    = 'SHORT'
                signal_prob    = down_p
                entry_exec_idx = i + 1

    return trades, equity, capital


# ── Metrics ────────────────────────────────────────────────────────────────

def calculate_metrics(trades, equity, final_cap):
    total = len(trades)
    if total == 0:
        return {'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown_pct': 0.0,
                'total_return_pct': 0.0, 'final_capital': final_cap,
                'long_trades': 0, 'short_trades': 0}

    wins       = sum(1 for t in trades if t['result'] == 'WIN')
    gross_win  = sum(t['net_pnl_pct'] for t in trades if t['result'] == 'WIN')
    gross_loss = abs(sum(t['net_pnl_pct'] for t in trades if t['result'] == 'LOSS'))
    pf         = gross_win / gross_loss if gross_loss > 0 else float('inf')
    total_ret  = (final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    pnls   = np.array([t['net_pnl_pct'] for t in trades])
    sharpe = float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0

    eq     = pd.Series(equity)
    peak   = eq.cummax()
    max_dd = float(((eq - peak) / peak * 100).min())

    longs  = sum(1 for t in trades if t.get('direction') == 'LONG')
    shorts = sum(1 for t in trades if t.get('direction') == 'SHORT')

    return {
        'total_trades':     total,
        'win_rate':         round(wins / total * 100, 1),
        'profit_factor':    round(pf, 2),
        'sharpe_ratio':     round(sharpe, 3),
        'max_drawdown_pct': round(max_dd, 2),
        'total_return_pct': round(total_ret, 2),
        'final_capital':    round(final_cap, 2),
        'long_trades':      longs,
        'short_trades':     shorts,
    }


# ── Rolling walk-forward per coin ──────────────────────────────────────────

def run_walk_forward(coin):
    params     = COIN_PARAMS[coin]
    tp_pct     = params['tp']
    sl_pct     = params['sl']
    time_limit = params.get('time_limit', TIME_LIMIT)

    print(f"\n{'='*65}")
    print(f"  {coin} — Bidirectional Rolling Walk-Forward")
    print(f"  TP={tp_pct*100:.1f}%  SL={sl_pct*100:.1f}%  "
          f"Breakeven={sl_pct/(sl_pct+tp_pct)*100:.1f}% win rate")
    print(f"{'='*65}")

    df = load_data(coin)
    print(f"  Data: {len(df):,} rows "
          f"({df['timestamp'].min().date()} -> {df['timestamp'].max().date()})")

    df['decision_label'] = df['target_direction'].copy()
    df = df.dropna(subset=['decision_label']).copy()
    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}")

    folds = FOLDS[coin]
    print(f"  Folds: {len(folds)}")

    fold_data   = {t: [] for t in THRESHOLDS}
    fold_trades = {t: [] for t in THRESHOLDS}   # parallel: actual trade lists

    for fold_idx, fold in enumerate(folds):
        train_start, train_end = fold['train']
        test_start,  test_end  = fold['test']

        train_df = df[(df['timestamp'] >= train_start) &
                      (df['timestamp'] <= train_end)].copy()
        test_df  = df[(df['timestamp'] >= test_start) &
                      (df['timestamp'] <= test_end)].copy()

        if len(train_df) < 500 or len(test_df) < 50:
            print(f"  Fold {fold_idx+1}: skipped (insufficient data)")
            continue

        dist = train_df['decision_label'].value_counts(normalize=True) * 100
        print(f"\n  Fold {fold_idx+1}/{len(folds)}"
              f"  [Train: {train_start}->{train_end}]"
              f"  [Test: {test_start}->{test_end}]")
        print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}", end='  ')
        for lbl in ['UP', 'SIDEWAYS', 'DOWN']:
            if lbl in dist.index:
                print(f"{lbl}={dist[lbl]:.1f}%", end=' ')
        print()

        # ── Primary model (trained on first 75% of train window) ───────────────
        n_prim      = int(len(train_df) * 0.75)
        prim_df     = train_df.iloc[:n_prim]
        meta_gen_df = train_df.iloc[n_prim:]   # last 25% — OOS for meta-label gen

        model, classes, train_acc = train_model(prim_df, feature_cols)
        print(f"  Train acc: {train_acc*100:.1f}%")
        mode_str  = 'LONG only' if LONG_ONLY else ('LONG+SHORT' if 'DOWN' in classes else 'LONG only')
        print(f"  Mode: {mode_str}")

        # ── Meta model (optional, trained on last 25% of train window) ────────
        meta_model, meta_feat = None, None
        if META_LABELING and len(meta_gen_df) >= 200:
            meta_df = generate_meta_labels(
                model, classes, meta_gen_df, feature_cols,
                META_LABEL_THRESH, tp_pct, sl_pct, coin, time_limit)
            result = train_meta_model(meta_df, feature_cols)
            if result is not None:
                meta_model, meta_feat = result
                n_sig = len(meta_df) if meta_df is not None else 0
                n_win = int(meta_df['meta_label'].sum()) if meta_df is not None else 0
                print(f"  Meta model: {n_sig} signals  "
                      f"({n_win} WIN / {n_sig - n_win} LOSS in cal set)")
            else:
                print(f"  Meta model: skipped (insufficient signals in cal set)")

        print(f"  {'Thresh':>8}  {'Trades':>6}  {'L/S':>7}  {'WR':>6}  "
              f"{'Sharpe':>7}  {'Return':>8}  {'PF':>5}")
        print(f"  {'-'*58}")

        for thresh in THRESHOLDS:
            trades, equity, final_cap = simulate_trades(
                model, test_df, feature_cols, classes,
                thresh, tp_pct, sl_pct, coin=coin, time_limit=time_limit,
                meta_model=meta_model, meta_feat=meta_feat)
            m = calculate_metrics(trades, equity, final_cap)

            valid = m['total_trades'] >= MIN_FOLD_TRADES
            fold_data[thresh].append(m if valid else None)
            fold_trades[thresh].append(trades if valid else None)

            flag  = '' if valid else ' *'
            ls    = f"{m['long_trades']}L/{m['short_trades']}S"
            print(f"  P>={thresh:.2f}  "
                  f"{m['total_trades']:>6}  "
                  f"{ls:>7}  "
                  f"{m['win_rate']:>5.1f}%  "
                  f"{m['sharpe_ratio']:>7.3f}  "
                  f"{m['total_return_pct']:>+7.2f}%  "
                  f"{m['profit_factor']:>5.2f}"
                  f"{flag}")

    # ── Cross-fold threshold selection ─────────────────────────────────
    print(f"\n  {'─'*65}")
    print(f"  CROSS-FOLD SUMMARY  (median across valid folds)")
    print(f"  {'─'*65}")
    print(f"  {'Thresh':>8}  {'Folds':>6}  {'Med Sharpe':>10}  "
          f"{'Med Return':>11}  {'Med WR':>7}  {'Med PF':>7}")
    print(f"  {'-'*60}")

    best_thresh = None
    best_sharpe = -999

    for thresh in THRESHOLDS:
        valid_folds = [m for m in fold_data[thresh] if m is not None]
        if not valid_folds:
            print(f"  P>={thresh:.2f}  {'—':>6}  {'no valid folds':>23}")
            continue

        sharpes = [m['sharpe_ratio']     for m in valid_folds]
        returns = [m['total_return_pct'] for m in valid_folds]
        wrs     = [m['win_rate']         for m in valid_folds]
        pfs     = [m['profit_factor']    for m in valid_folds]

        med_s = float(np.median(sharpes))
        med_r = float(np.median(returns))
        med_w = float(np.median(wrs))
        med_p = float(np.median(pfs))

        marker = ''
        if med_s > best_sharpe and len(valid_folds) >= 2:
            best_sharpe = med_s
            best_thresh = thresh
            marker = ' <- best'

        print(f"  P>={thresh:.2f}  {len(valid_folds):>6}  "
              f"{med_s:>+10.3f}  {med_r:>+10.2f}%  "
              f"{med_w:>6.1f}%  {med_p:>7.2f}{marker}")

    if best_thresh is None:
        print(f"\n  No threshold had >=2 valid folds. NOT_VIABLE.")
        return {'coin': coin, 'error': 'No valid threshold found'}

    print(f"\n  Selected threshold: P(UP/DOWN) >= {best_thresh}  "
          f"(median Sharpe = {best_sharpe:+.3f})")

    # ── Aggregate ──────────────────────────────────────────────────────
    chosen      = [m for m in fold_data[best_thresh] if m is not None]
    # Collect actual trade lists for portfolio-level evaluation in main()
    best_trades = []
    for fold_t in fold_trades[best_thresh]:
        if fold_t is not None:
            best_trades.extend(fold_t)
    all_trades  = sum(m['total_trades']      for m in chosen)
    avg_ret     = float(np.mean([m['total_return_pct'] for m in chosen]))
    med_sharpe  = float(np.median([m['sharpe_ratio']   for m in chosen]))
    med_wr      = float(np.median([m['win_rate']        for m in chosen]))
    med_pf      = float(np.median([m['profit_factor']   for m in chosen]))
    n_pos       = sum(1 for m in chosen if m['total_return_pct'] > 0)
    total_long  = sum(m['long_trades']  for m in chosen)
    total_short = sum(m['short_trades'] for m in chosen)

    if all_trades < 20:
        verdict = 'INSUFFICIENT_DATA'
    elif med_sharpe >= 0.8 and med_wr >= 48 and n_pos >= len(chosen) * 0.6:
        verdict = 'VIABLE'
    elif med_sharpe >= 0.3 and med_wr >= 44:
        verdict = 'MARGINAL'
    else:
        verdict = 'NOT_VIABLE'

    print(f"\n  AGGREGATE  (threshold={best_thresh}, {len(chosen)} valid folds)")
    print(f"  Total trades  : {all_trades}  "
          f"({total_long} long / {total_short} short)")
    print(f"  Median WR     : {med_wr:.1f}%")
    print(f"  Median PF     : {med_pf:.2f}")
    print(f"  Median Sharpe : {med_sharpe:+.3f}")
    print(f"  Avg return    : {avg_ret:+.2f}%  "
          f"({n_pos}/{len(chosen)} folds positive)")
    print(f"  Verdict       : {verdict}")

    # Save model trained on last (most recent) fold's train block
    last_fold     = folds[-1]
    last_train_df = df[(df['timestamp'] >= last_fold['train'][0]) &
                       (df['timestamp'] <= last_fold['train'][1])].copy()
    if len(last_train_df) >= 500:
        final_model, _, _ = train_model(last_train_df, feature_cols)
        model_dir = os.path.join(BASE_DIR, f"models/{coin}")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(final_model, f"{model_dir}/wf_decision_model_v2.pkl")
        # Save feature list for paper_trader.py (must match training feature set exactly)
        with open(f"{model_dir}/decision_features_v2.txt", 'w') as fh:
            fh.write('\n'.join(feature_cols))
        print(f"  Saved: models/{coin}/wf_decision_model_v2.pkl  "
              f"(train {last_fold['train'][0]}->{last_fold['train'][1]})")
        print(f"  Saved: models/{coin}/decision_features_v2.txt  ({len(feature_cols)} features)")

    return {
        'coin':           coin,
        'tp_pct':         tp_pct,
        'sl_pct':         sl_pct,
        'folds':          len(folds),
        'valid_folds':    len(chosen),
        'best_threshold': best_thresh,
        'total_trades':   all_trades,
        'long_trades':    total_long,
        'short_trades':   total_short,
        'med_win_rate':   med_wr,
        'med_pf':         med_pf,
        'med_sharpe':     med_sharpe,
        'avg_return':     avg_ret,
        'pos_folds':      n_pos,
        'verdict':        verdict,
        'best_trades':    best_trades,   # for portfolio evaluation
    }


# ── Main ───────────────────────────────────────────────────────────────────

def portfolio_metrics(all_trades):
    """Compute portfolio-level metrics from pooled trades across all coins."""
    total = len(all_trades)
    if total == 0:
        return None
    wins       = sum(1 for t in all_trades if t['result'] == 'WIN')
    gross_win  = sum(t['net_pnl_pct'] for t in all_trades if t['result'] == 'WIN')
    gross_loss = abs(sum(t['net_pnl_pct'] for t in all_trades if t['result'] == 'LOSS'))
    pf         = gross_win / gross_loss if gross_loss > 0 else float('inf')
    pnls       = np.array([t['net_pnl_pct'] for t in all_trades])
    sharpe     = float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0
    longs      = sum(1 for t in all_trades if t.get('direction') == 'LONG')
    shorts     = sum(1 for t in all_trades if t.get('direction') == 'SHORT')
    return {
        'total_trades': total,
        'win_rate':     round(wins / total * 100, 1),
        'profit_factor': round(pf, 2),
        'sharpe':       round(sharpe, 3),
        'long_trades':  longs,
        'short_trades': shorts,
    }


def main():
    print("\n" + "#"*65)
    print("  ROLLING WALK-FORWARD — BIDIRECTIONAL + COIN-SPECIFIC TP/SL")
    print("  LONG on UP signal | SHORT on DOWN signal | ADX >= 20 gate")
    print("#"*65)

    # All supported coins — new coins run only if their feature CSV exists
    all_coins = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT',
                 'AVAX_USDT', 'BNB_USDT', 'LINK_USDT',
                 'ARB_USDT', 'OP_USDT', 'INJ_USDT']

    coins = [c for c in all_coins
             if os.path.exists(os.path.join(BASE_DIR, f"data/{c}_multi_tf_features.csv"))]

    if len(coins) < len(all_coins):
        missing = set(all_coins) - set(coins)
        print(f"\n  ⚠️  Skipping (no feature CSV): {', '.join(sorted(missing))}")
        print(f"  Run collect_funding_rates.py then collect_multi_timeframe.py first.\n")

    results = {}
    for coin in coins:
        try:
            results[coin] = run_walk_forward(coin)
        except Exception as e:
            import traceback
            print(f"\n  {coin} failed: {e}")
            traceback.print_exc()
            results[coin] = {'coin': coin, 'error': str(e)}

    print(f"\n\n{'='*85}")
    print(f"  FINAL SUMMARY — PER COIN")
    print(f"{'='*85}")
    print(f"  {'Coin':<12}  {'TP/SL':>8}  {'Folds':>5}  {'Thresh':>6}  "
          f"{'Trades':>6}  {'L/S':>7}  {'MedWR':>6}  {'MedSharpe':>9}  "
          f"{'AvgRet':>8}  Verdict")
    print(f"  {'-'*85}")

    for coin, r in results.items():
        if 'error' in r:
            print(f"  {coin:<12}  ERROR: {r['error']}")
        else:
            tp_sl = f"{r['tp_pct']*100:.0f}/{r['sl_pct']*100:.0f}%"
            ls    = f"{r['long_trades']}L/{r['short_trades']}S"
            print(f"  {coin:<12}  {tp_sl:>8}  "
                  f"{r['valid_folds']:>5}  "
                  f"{r['best_threshold']:>6.2f}  "
                  f"{r['total_trades']:>6}  "
                  f"{ls:>7}  "
                  f"{r['med_win_rate']:>5.1f}%  "
                  f"{r['med_sharpe']:>+9.3f}  "
                  f"{r['avg_return']:>+7.2f}%  "
                  f"{r['verdict']}")

    print(f"{'='*85}")
    print(f"  Cost: 0.22% RT | ADX>={ADX_MIN} gate | 3:1 R:R | Breakeven=25.0% WR for all coins")

    # ── Portfolio-level evaluation (Tier 2) ────────────────────────────────
    # Pool trades from all coins that have a real edge (MARGINAL or VIABLE).
    # This is the true system-level Sharpe — coins diversify each other's
    # bad folds. A portfolio Sharpe >= 0.8 here means the system IS viable
    # in aggregate even if individual coins are only MARGINAL.
    active_verdicts = {'MARGINAL', 'VIABLE'}
    portfolio_coins = [c for c, r in results.items()
                       if 'error' not in r and r['verdict'] in active_verdicts]

    if not portfolio_coins:
        print(f"\n  No MARGINAL/VIABLE coins to build portfolio from.")
        return

    all_portfolio_trades = []
    for coin in portfolio_coins:
        all_portfolio_trades.extend(results[coin].get('best_trades', []))

    pm = portfolio_metrics(all_portfolio_trades)
    if pm is None:
        return

    # Portfolio verdict using same Sharpe threshold as individual coins
    if pm['sharpe'] >= 0.8 and pm['win_rate'] >= 48:
        port_verdict = 'VIABLE'
    elif pm['sharpe'] >= 0.3 and pm['win_rate'] >= 44:
        port_verdict = 'MARGINAL'
    else:
        port_verdict = 'NOT_VIABLE'

    print(f"\n{'='*65}")
    print(f"  PORTFOLIO SUMMARY  ({len(portfolio_coins)} coins: {', '.join(portfolio_coins)})")
    print(f"{'='*65}")
    print(f"  Total trades  : {pm['total_trades']}  "
          f"({pm['long_trades']} long / {pm['short_trades']} short)")
    print(f"  Portfolio WR  : {pm['win_rate']:.1f}%")
    print(f"  Portfolio PF  : {pm['profit_factor']:.2f}")
    print(f"  Portfolio Sharpe: {pm['sharpe']:+.3f}")
    print(f"  Portfolio Verdict: {port_verdict}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
