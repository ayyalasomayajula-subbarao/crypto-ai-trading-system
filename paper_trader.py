"""
Live Paper Trading Engine
==========================
Runs as a background asyncio task inside the FastAPI server.
Processes each 1h candle for SOL_USDT and PEPE_USDT using
frozen walk-forward validated models.

NO retraining. NO parameter changes. NO manual overrides.
This is a scientific experiment.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings('ignore')

# Numpy 2.x compatibility: models pickled with numpy 1.x reference numpy.core
import numpy.core.multiarray  # noqa: F401  ensure it exists
import sys
if not hasattr(np, '_core'):
    # numpy 1.x — alias core as _core for pickles saved on numpy 2.x
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
elif not hasattr(np, 'core'):
    # numpy 2.x — alias _core as core for pickles saved on numpy 1.x
    sys.modules['numpy.core'] = np._core
    sys.modules['numpy.core.multiarray'] = np._core.multiarray

from ta.trend import SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "data", "paper_trading_state.json")
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

# ── Frozen configuration — update after each WF run ──────────────
# WF-validated thresholds (90-feature model, META_LABELING=True):
#   BTC thresh=0.50, LINK thresh=0.55 (from 90-feature WF run)
# Coin-specific TP/SL from walk_forward_validation.py COIN_PARAMS.
# Regime gate: 1w_dist_sma_50 for all except PEPE (uses 1d_dist_sma_50).
CONFIG = {
    # WF-validated thresholds (90-feature, META_LABELING=True, n_estimators=200 n_jobs=1):
    #   BTC thresh=0.50 — Sharpe +0.382, WR 57.1%, 3 folds, 43 trades
    #   LINK thresh=0.55 — Sharpe +0.456, WR 50.0%, 3 folds, 21 trades
    'coins': {
        'BTC_USDT':  {'threshold': 0.50, 'tp': 0.030, 'sl': 0.015, 'time_limit': 48},
        'LINK_USDT': {'threshold': 0.55, 'tp': 0.075, 'sl': 0.025, 'time_limit': 72},
    },
    'adx_min': 20,
    'regime_gate': True,      # 1w_dist_sma_50 > 0 for LONG; < 0 for SHORT
    'risk_per_trade': 0.005,  # 0.5% equity risk per trade
    'fee_pct': 0.0006,
    'slippage_pct': 0.0003,
    'spread_pct': 0.0002,
    'initial_capital': 10000,
    'check_interval_seconds': 60,
}
CONFIG['round_trip_cost'] = (CONFIG['fee_pct'] + CONFIG['slippage_pct'] + CONFIG['spread_pct']) * 2


def add_features_for_tf(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Compute technical features per timeframe — must match collect_multi_timeframe.py exactly.
    90-feature format: sma_21/50 + dist + slopes; rsi; macd_diff only; bb; atr_pct; adx;
    momentum; volume_ratio; dist_from_10p_high. 1H only: realized vol features.
    """
    features = pd.DataFrame(index=data.index)

    features[f'{prefix}_returns_1'] = data['close'].pct_change(1) * 100
    features[f'{prefix}_returns_5'] = data['close'].pct_change(5) * 100
    features[f'{prefix}_hl_range'] = (data['high'] - data['low']) / data['close'] * 100

    # SMA 21 and 50 only (sma_7/14 and all EMAs removed — low importance, redundant)
    for period in [21, 50]:
        sma = SMAIndicator(data['close'], period).sma_indicator()
        features[f'{prefix}_sma_{period}'] = sma
        features[f'{prefix}_dist_sma_{period}'] = (data['close'] - sma) / data['close'] * 100

    # SMA slope (% change over last 5 bars) — new high-importance feature
    sma21 = features[f'{prefix}_sma_21']
    sma50 = features[f'{prefix}_sma_50']
    features[f'{prefix}_sma21_slope'] = sma21.pct_change(5) * 100
    features[f'{prefix}_sma50_slope'] = sma50.pct_change(5) * 100

    features[f'{prefix}_rsi'] = RSIIndicator(data['close'], 14).rsi()

    # MACD diff only (macd/macd_signal removed — redundant with diff)
    features[f'{prefix}_macd_diff'] = MACD(data['close']).macd_diff()

    bb = BollingerBands(data['close'], 20, 2)
    features[f'{prefix}_bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    features[f'{prefix}_bb_position'] = (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

    # ATR % only (raw ATR is price-scale-dependent)
    _atr = AverageTrueRange(data['high'], data['low'], data['close'], 14).average_true_range()
    features[f'{prefix}_atr_pct'] = _atr / data['close'] * 100

    try:
        features[f'{prefix}_adx'] = ADXIndicator(data['high'], data['low'], data['close'], 14).adx()
    except Exception:
        features[f'{prefix}_adx'] = 25.0

    features[f'{prefix}_momentum'] = data['close'] / data['close'].shift(10) - 1

    # Volume ratio (volume_ma removed — only ratio matters)
    _vol_ma = data['volume'].rolling(14).mean()
    features[f'{prefix}_volume_ratio'] = data['volume'] / _vol_ma.clip(lower=1e-8)

    # Pullback depth — % below recent 10-period high
    _rolling_high = data['high'].rolling(10).max()
    features[f'{prefix}_dist_from_10p_high'] = (data['close'] - _rolling_high) / _rolling_high * 100

    # 1H-only realized volatility features
    if prefix == '1h':
        log_ret = np.log(data['close'] / data['close'].shift(1))
        features['1h_realized_vol_24h'] = log_ret.rolling(24).std() * np.sqrt(24) * 100
        features['1h_realized_vol_7d']  = log_ret.rolling(168).std() * np.sqrt(168) * 100
        vol_24h = features['1h_realized_vol_24h']
        vol_7d  = features['1h_realized_vol_7d']
        features['1h_vol_regime_ratio'] = vol_24h / vol_7d.clip(lower=1e-8)

    features['timestamp'] = data['timestamp']
    return features


def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch OHLCV + taker buy volume from Binance REST API (no auth needed).
    Returns: timestamp, open, high, low, close, volume, taker_buy_volume
    taker_sell_volume = volume - taker_buy_volume
    """
    binance_symbol = symbol.replace('_', '')
    params = {'symbol': binance_symbol, 'interval': interval, 'limit': limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']:
        df[col] = df[col].astype(float)
    df['taker_buy_volume'] = df['taker_buy_base']

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_volume']]


def fetch_latest_funding_rate(coin: str, limit: int = 50) -> dict:
    """
    Fetch recent funding rate history from Binance fapi.
    Returns dict with funding_rate, funding_rate_3d_avg, funding_rate_7d_avg,
    funding_rate_momentum, funding_trend (latest values).
    """
    symbol_map = {
        'BTC_USDT': 'BTCUSDT', 'ETH_USDT': 'ETHUSDT', 'SOL_USDT': 'SOLUSDT',
        'PEPE_USDT': '1000PEPEUSDT', 'AVAX_USDT': 'AVAXUSDT', 'BNB_USDT': 'BNBUSDT',
        'LINK_USDT': 'LINKUSDT', 'ARB_USDT': 'ARBUSDT', 'OP_USDT': 'OPUSDT',
        'INJ_USDT': 'INJUSDT',
    }
    symbol = symbol_map.get(coin, coin.replace('_', ''))
    try:
        resp = requests.get(
            BINANCE_FUNDING_URL,
            params={'symbol': symbol, 'limit': limit},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return {}
        rates = pd.Series([float(r['fundingRate']) for r in data])
        # Funding is 8h; 3d = last 9 records; 7d = last 21
        return {
            'funding_rate':          float(rates.iloc[-1]),
            'funding_rate_3d_avg':   float(rates.iloc[-9:].mean()),
            'funding_rate_7d_avg':   float(rates.iloc[-21:].mean()),
            'funding_rate_momentum': float(rates.iloc[-1] - rates.iloc[-9]) if len(rates) >= 9 else 0.0,
            'funding_trend':         float((rates.iloc[-1] - rates.iloc[-9:].mean()) * 3) if len(rates) >= 9 else 0.0,
        }
    except Exception as e:
        print(f"  [PAPER] Funding rate fetch error for {coin}: {e}")
        return {}


def compute_live_features(coin: str) -> Optional[pd.Series]:
    """
    Fetch latest candles for all timeframes + funding rates + taker volume.
    Computes 82 ML features matching collect_multi_timeframe.py exactly.
    Lookahead fix: 4H/1D/1W use PREVIOUS completed candle (floor-subtract).
    """
    try:
        # ── Fetch OHLCV + taker buy volume ────────────────────────────────
        df_1h = fetch_klines(coin, '1h', 250)   # ~10 days
        df_4h = fetch_klines(coin, '4h', 100)   # ~17 days
        df_1d = fetch_klines(coin, '1d', 100)   # ~3 months
        df_1w = fetch_klines(coin, '1w', 60)    # ~14 months

        # ── Compute per-TF features ────────────────────────────────────────
        feat_1h = add_features_for_tf(df_1h, '1h')
        feat_4h = add_features_for_tf(df_4h, '4h')
        feat_1d = add_features_for_tf(df_1d, '1d')
        feat_1w = add_features_for_tf(df_1w, '1w')

        # ── Normalise timestamp precision to ms (avoids merge dtype mismatches) ──
        def _to_ms(ts: pd.Series) -> pd.Series:
            return ts.astype('datetime64[ms]')

        for feat_df in [feat_1h, feat_4h, feat_1d, feat_1w]:
            feat_df['timestamp'] = _to_ms(feat_df['timestamp'])

        # ── Start with 1h base ────────────────────────────────────────────
        df = df_1h[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df['timestamp'] = _to_ms(df['timestamp'])
        df = df.merge(feat_1h.drop(columns=['timestamp']), left_index=True, right_index=True)

        # ── Merge 4H: PREVIOUS completed candle (floor - 4h) ──────────────
        feat_4h['timestamp_4h'] = _to_ms(feat_4h['timestamp'].dt.floor('4h'))
        df['timestamp_4h']      = _to_ms(df['timestamp'].dt.floor('4h') - pd.Timedelta(hours=4))
        df = pd.merge_asof(
            df.sort_values('timestamp_4h'),
            feat_4h.drop(columns=['timestamp']).sort_values('timestamp_4h'),
            on='timestamp_4h', direction='backward'
        ).sort_values('timestamp')
        df.drop(columns=['timestamp_4h'], inplace=True)

        # ── Merge 1D: PREVIOUS completed candle (floor - 1d) ──────────────
        feat_1d['timestamp_1d'] = _to_ms(feat_1d['timestamp'].dt.floor('1d'))
        df['timestamp_1d']      = _to_ms(df['timestamp'].dt.floor('1d') - pd.Timedelta(days=1))
        df = pd.merge_asof(
            df.sort_values('timestamp_1d'),
            feat_1d.drop(columns=['timestamp']).sort_values('timestamp_1d'),
            on='timestamp_1d', direction='backward'
        ).sort_values('timestamp')
        df.drop(columns=['timestamp_1d'], inplace=True)

        # ── Merge 1W: PREVIOUS completed week (floor week - 7d) ───────────
        feat_1w['timestamp_1w'] = _to_ms(feat_1w['timestamp'].dt.to_period('W').dt.start_time)
        df['timestamp_1w']      = _to_ms(
            df['timestamp'].dt.to_period('W').dt.start_time - pd.Timedelta(weeks=1)
        )
        df = pd.merge_asof(
            df.sort_values('timestamp_1w'),
            feat_1w.drop(columns=['timestamp']).sort_values('timestamp_1w'),
            on='timestamp_1w', direction='backward'
        ).sort_values('timestamp').reset_index(drop=True)
        df.drop(columns=['timestamp_1w'], inplace=True)

        # ── Funding rate features ──────────────────────────────────────────
        funding = fetch_latest_funding_rate(coin)
        for k, v in funding.items():
            df[k] = v

        # ── Taker order flow features ──────────────────────────────────────
        _vol = df_1h['volume'].clip(lower=1e-8)
        _buy = df_1h['taker_buy_volume']
        _imb = (_buy - (_vol - _buy)) / _vol
        df_tv = df_1h[['timestamp']].copy()
        df_tv['imbalance_4h_ma']  = _imb.rolling(4,  min_periods=2).mean().values
        df_tv['imbalance_24h_ma'] = _imb.rolling(24, min_periods=8).mean().values
        df = pd.merge_asof(
            df.sort_values('timestamp'),
            df_tv.sort_values('timestamp'),
            on='timestamp', direction='backward'
        )

        # ── Drop rows with insufficient warmup ────────────────────────────
        drop_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume',
                     'taker_buy_volume', 'target_return', 'target_direction', 'decision_label'}
        feature_cols = [c for c in df.columns if c not in drop_cols]
        df_clean = df[feature_cols + ['timestamp', 'open', 'high', 'low', 'close']].dropna(
            subset=feature_cols
        )

        if len(df_clean) == 0:
            return None

        return df_clean.iloc[-1]

    except Exception as e:
        print(f"  [PAPER] Feature computation error for {coin}: {e}")
        import traceback
        traceback.print_exc()
        return None


class PaperTrader:
    """Live paper trading engine with persistent state."""

    def __init__(self):
        self.running = False
        self.models: Dict = {}
        self.feature_cols: Dict[str, List[str]] = {}
        self.state: Dict = {}
        self._task: Optional[asyncio.Task] = None
        self._last_candle_ts: Dict[str, str] = {}

    def load_models(self):
        """Load frozen walk-forward 3-class (UP/DOWN/SIDEWAYS) models."""
        for coin in CONFIG['coins']:
            model_path = os.path.join(BASE_DIR, f"models/{coin}/wf_decision_model_v2.pkl")
            features_path = os.path.join(BASE_DIR, f"models/{coin}/decision_features_v2.txt")

            if not os.path.exists(model_path):
                print(f"  [PAPER] WARNING: Model not found: {model_path}")
                continue

            model = joblib.load(model_path)
            classes = list(model.classes_)

            # Load saved feature list (written by WF after training)
            feature_cols = None
            if os.path.exists(features_path):
                with open(features_path) as f:
                    feature_cols = [line.strip() for line in f if line.strip()]
                print(f"  [PAPER] Feature list: {len(feature_cols)} features from {features_path}")
            else:
                print(f"  [PAPER] WARNING: No feature list found at {features_path}")

            self.models[coin] = {
                'model':    model,
                'classes':  classes,
                'up_idx':   classes.index('UP')   if 'UP'   in classes else None,
                'down_idx': classes.index('DOWN') if 'DOWN' in classes else None,
            }
            self.feature_cols[coin] = feature_cols  # None = derive from live features
            print(f"  [PAPER] Loaded model: {coin} | classes={classes}")

    def load_state(self):
        """Load persistent state from disk."""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                self.state = json.load(f)
            print(f"  [PAPER] Resumed state: {len(self.state.get('trades', []))} trades, "
                  f"equity=${self.state.get('equity', CONFIG['initial_capital']):,.2f}")
        else:
            self.state = {
                'start_time': datetime.now(timezone.utc).isoformat(),
                'equity': CONFIG['initial_capital'],
                'initial_capital': CONFIG['initial_capital'],
                'positions': {},  # coin -> position dict
                'trades': [],
                'equity_curve': [{'timestamp': datetime.now(timezone.utc).isoformat(),
                                  'equity': CONFIG['initial_capital']}],
                'config': {
                    'coins': list(CONFIG['coins'].keys()),
                    'thresholds': {c: cfg['threshold'] for c, cfg in CONFIG['coins'].items()},
                    'tp_sl': {c: {'tp': cfg['tp'], 'sl': cfg['sl'], 'time_limit': cfg['time_limit']}
                              for c, cfg in CONFIG['coins'].items()},
                    'risk_per_trade': CONFIG['risk_per_trade'],
                    'round_trip_cost': CONFIG['round_trip_cost'],
                },
                'last_processed': {},
            }
            self.save_state()

    def save_state(self):
        """Persist state to disk."""
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def start(self, capital: float = None):
        """Start the paper trading loop."""
        if self.running:
            return {'status': 'already_running'}

        self.load_models()
        self.load_state()

        # Allow setting initial capital on first start (before any trades)
        if capital and capital > 0 and not self.state.get('trades'):
            self.state['equity'] = capital
            self.state['initial_capital'] = capital
            self.state['equity_curve'] = [{'timestamp': datetime.now(timezone.utc).isoformat(),
                                           'equity': capital}]
            self.save_state()

        self.running = True
        self._task = asyncio.ensure_future(self._run_loop())
        print(f"  [PAPER] Started paper trading: {list(CONFIG['coins'].keys())}")
        return {'status': 'started', 'coins': list(CONFIG['coins'].keys())}

    def stop(self):
        """Stop the paper trading loop."""
        self.running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self.save_state()
        print("  [PAPER] Stopped paper trading")
        return {'status': 'stopped'}

    async def _run_loop(self):
        """Main loop: check for new candles every minute."""
        print(f"  [PAPER] Loop started, checking every {CONFIG['check_interval_seconds']}s")

        while self.running:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._process_tick)
            except Exception as e:
                print(f"  [PAPER] Tick error: {e}")

            await asyncio.sleep(CONFIG['check_interval_seconds'])

    def _process_tick(self):
        """Process one tick: fetch data, check signals, manage positions."""
        for coin in CONFIG['coins']:
            if coin not in self.models:
                continue

            try:
                # Fetch latest 1h candle
                df_1h = fetch_klines(coin, '1h', 3)
                if len(df_1h) < 2:
                    continue

                # Use second-to-last candle (last CLOSED candle)
                latest_closed = df_1h.iloc[-2]
                candle_ts = latest_closed['timestamp'].isoformat()

                # Skip if already processed this candle
                if self.state['last_processed'].get(coin) == candle_ts:
                    continue

                self.state['last_processed'][coin] = candle_ts
                now_str = datetime.now(timezone.utc).isoformat()

                # Check exit for open position
                if coin in self.state['positions']:
                    self._check_exit(coin, latest_closed, now_str)

                # Check entry signal (only if no position)
                if coin not in self.state['positions']:
                    self._check_entry(coin, now_str)

            except Exception as e:
                print(f"  [PAPER] Error processing {coin}: {e}")

        # Update equity curve (daily snapshot)
        self._update_equity_curve()
        self.save_state()

    def _check_entry(self, coin: str, now_str: str):
        """
        Check if model generates entry signal — bidirectional (LONG or SHORT).
        Gates: P(UP/DOWN) >= threshold + ADX >= 20 + regime gate (1w_dist_sma_50).
        Coin-specific TP/SL from CONFIG.
        """
        features = compute_live_features(coin)
        if features is None:
            return

        model_info = self.models[coin]
        threshold = CONFIG['coins'][coin]['threshold']
        tp_pct    = CONFIG['coins'][coin]['tp']
        sl_pct    = CONFIG['coins'][coin]['sl']

        # ADX gate
        adx = float(features.get('1h_adx', 0))
        if adx < CONFIG['adx_min']:
            return

        # Build feature vector for prediction
        feature_cols = self.feature_cols[coin]
        try:
            if feature_cols is not None:
                X = pd.DataFrame([[features.get(col, 0) for col in feature_cols]], columns=feature_cols)
            else:
                # Derive feature set dynamically (exclude metadata columns)
                drop = {'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'taker_buy_volume', 'target_return', 'target_direction', 'decision_label'}
                cols = [c for c in features.index if c not in drop]
                X = pd.DataFrame([[features.get(c, 0) for c in cols]], columns=cols)
            X = X.fillna(0)
        except Exception as e:
            print(f"  [PAPER] Feature extraction error {coin}: {e}")
            return

        # Predict probabilities
        probas = model_info['model'].predict_proba(X)[0]
        total = probas.sum()
        if total > 0:
            probas = probas / total

        up_idx   = model_info['up_idx']
        down_idx = model_info['down_idx']
        up_prob   = float(probas[up_idx])   if up_idx   is not None else 0.0
        down_prob = float(probas[down_idx]) if down_idx is not None else 0.0

        # Regime gate: 1w_dist_sma_50 > 0 = above weekly MA (bull bias)
        regime_dist = float(features.get('1w_dist_sma_50', 0))
        if coin == 'PEPE_USDT':
            regime_dist = float(features.get('1d_dist_sma_50', 0))

        go_long  = up_prob   >= threshold
        go_short = down_prob >= threshold

        if CONFIG['regime_gate']:
            if go_long  and regime_dist <= 0:
                go_long  = False
            if go_short and regime_dist >= 0:
                go_short = False

        # If both fire, take stronger signal
        if go_long and go_short:
            go_long  = up_prob >= down_prob
            go_short = not go_long

        if not go_long and not go_short:
            return

        direction = 'LONG' if go_long else 'SHORT'
        entry_price = float(features.get('close', 0))
        if entry_price <= 0:
            return

        # Position sizing: risk_per_trade % of equity / SL distance
        equity = self.state['equity']
        risk_amount = equity * CONFIG['risk_per_trade']
        position_size_usd = risk_amount / sl_pct

        if direction == 'LONG':
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)

        position = {
            'coin':              coin,
            'direction':         direction,
            'entry_time':        now_str,
            'entry_price':       entry_price,
            'tp_price':          tp_price,
            'sl_price':          sl_price,
            'position_size_usd': round(position_size_usd, 2),
            'up_prob':           round(up_prob * 100, 1),
            'down_prob':         round(down_prob * 100, 1),
            'adx':               round(adx, 1),
            'time_limit':        CONFIG['coins'][coin]['time_limit'],
            'candles_held':      0,
        }
        self.state['positions'][coin] = position
        print(f"  [PAPER] ENTRY {coin} {direction}: ${entry_price:.4f} | "
              f"TP=${tp_price:.4f} SL=${sl_price:.4f} | "
              f"Size=${position_size_usd:.2f} | "
              f"P(UP)={up_prob*100:.1f}% P(DOWN)={down_prob*100:.1f}% ADX={adx:.0f}")

    def _check_exit(self, coin: str, candle: pd.Series, now_str: str):
        """
        Check if open position should be closed — handles both LONG and SHORT.
        LONG:  TP hit if high >= tp_price; SL hit if low  <= sl_price
        SHORT: TP hit if low  <= tp_price; SL hit if high >= sl_price
        Worst-case rule: if both hit on same candle, SL wins.
        """
        pos = self.state['positions'][coin]
        pos['candles_held'] += 1

        high        = float(candle['high'])
        low         = float(candle['low'])
        close       = float(candle['close'])
        direction   = pos.get('direction', 'LONG')
        entry_price = pos['entry_price']
        tp_price    = pos['tp_price']
        sl_price    = pos['sl_price']
        time_limit  = pos.get('time_limit', CONFIG['coins'].get(coin, {}).get('time_limit', 48))

        exit_reason = None
        exit_price  = None

        if direction == 'LONG':
            tp_hit = high >= tp_price
            sl_hit = low  <= sl_price
        else:  # SHORT
            tp_hit = low  <= tp_price
            sl_hit = high >= sl_price

        if tp_hit and sl_hit:
            exit_reason = 'SL'   # worst-case: SL wins
            exit_price  = sl_price
        elif sl_hit:
            exit_reason = 'SL'
            exit_price  = sl_price
        elif tp_hit:
            exit_reason = 'TP'
            exit_price  = tp_price
        elif pos['candles_held'] >= time_limit:
            exit_reason = 'TIME'
            exit_price  = close

        if exit_reason:
            if direction == 'LONG':
                gross_pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                gross_pnl_pct = (entry_price - exit_price) / entry_price * 100

            net_pnl_pct = gross_pnl_pct - (CONFIG['round_trip_cost'] * 100)
            pnl_usd = pos['position_size_usd'] * (net_pnl_pct / 100)
            self.state['equity'] += pnl_usd

            trade = {
                'coin':              coin,
                'direction':         direction,
                'entry_time':        pos['entry_time'],
                'exit_time':         now_str,
                'entry_price':       round(entry_price, 8),
                'exit_price':        round(exit_price, 8),
                'position_size_usd': pos['position_size_usd'],
                'up_prob':           pos.get('up_prob', 0),
                'down_prob':         pos.get('down_prob', 0),
                'adx':               pos.get('adx', 0),
                'gross_pnl_pct':     round(gross_pnl_pct, 4),
                'net_pnl_pct':       round(net_pnl_pct, 4),
                'pnl_usd':           round(pnl_usd, 2),
                'exit_reason':       exit_reason,
                'hours_held':        pos['candles_held'],
                'result':            'WIN' if net_pnl_pct > 0 else 'LOSS',
                'equity_after':      round(self.state['equity'], 2),
            }
            self.state['trades'].append(trade)
            del self.state['positions'][coin]

            print(f"  [PAPER] EXIT {coin} {direction}: {exit_reason} | "
                  f"PnL={net_pnl_pct:+.2f}% (${pnl_usd:+.2f}) | "
                  f"Equity=${self.state['equity']:,.2f} | "
                  f"Held {pos['candles_held']}h")

    def _update_equity_curve(self):
        """Add daily equity snapshot."""
        curve = self.state['equity_curve']
        now = datetime.now(timezone.utc)

        # Only add once per hour
        if curve:
            last_ts = datetime.fromisoformat(curve[-1]['timestamp'].replace('Z', '+00:00'))
            if (now - last_ts).total_seconds() < 3500:
                return

        curve.append({
            'timestamp': now.isoformat(),
            'equity': round(self.state['equity'], 2),
        })

    def get_status(self) -> Dict:
        """Return current paper trading status."""
        trades = self.state.get('trades', [])
        start_time = self.state.get('start_time', '')

        # Calculate days running
        days_running = 0
        if start_time:
            try:
                start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                days_running = (datetime.now(timezone.utc) - start).days
            except Exception:
                pass

        init_cap = self.state.get('initial_capital', CONFIG['initial_capital'])
        cur_equity = self.state.get('equity', init_cap)

        return {
            'running': self.running,
            'start_time': start_time,
            'days_running': days_running,
            'target_days': 45,
            'equity': round(cur_equity, 2),
            'initial_capital': init_cap,
            'total_return_pct': round(
                (cur_equity - init_cap) / init_cap * 100, 2
            ) if init_cap > 0 else 0,
            'total_trades': len(trades),
            'open_positions': self.state.get('positions', {}),
            'coins': list(CONFIG['coins'].keys()),
            'config': self.state.get('config', {}),
            'last_processed': self.state.get('last_processed', {}),
        }

    def get_trades(self) -> List[Dict]:
        """Return all trades."""
        return self.state.get('trades', [])

    def get_metrics(self) -> Dict:
        """Calculate comprehensive metrics."""
        trades = self.state.get('trades', [])
        equity_curve = self.state.get('equity_curve', [])

        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown_pct': 0,
                'total_return_pct': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'per_coin': {},
            }

        total = len(trades)
        wins = [t for t in trades if t['result'] == 'WIN']
        losses = [t for t in trades if t['result'] == 'LOSS']

        win_rate = len(wins) / total * 100 if total > 0 else 0
        gross_profit = sum(t['pnl_usd'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl_usd'] for t in losses)) if losses else 0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.99

        avg_win = np.mean([t['net_pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['net_pnl_pct'] for t in losses]) if losses else 0

        # Sharpe from equity curve
        sharpe = 0.0
        if len(equity_curve) > 48:
            eq = pd.Series([p['equity'] for p in equity_curve])
            daily_eq = eq.iloc[::24]
            if len(daily_eq) > 2:
                daily_ret = daily_eq.pct_change().dropna()
                if daily_ret.std() > 0:
                    sharpe = round(float((daily_ret.mean() / daily_ret.std()) * np.sqrt(365)), 2)

        # Max drawdown
        max_dd = 0.0
        if equity_curve:
            eq = pd.Series([p['equity'] for p in equity_curve])
            peak = eq.cummax()
            dd = (eq - peak) / peak * 100
            max_dd = round(float(dd.min()), 2)

        init_cap = self.state.get('initial_capital', CONFIG['initial_capital'])
        total_return = round(
            (self.state.get('equity', init_cap) - init_cap)
            / init_cap * 100, 2
        ) if init_cap > 0 else 0

        # Per-coin breakdown
        per_coin = {}
        for coin in CONFIG['coins']:
            coin_trades = [t for t in trades if t['coin'] == coin]
            if coin_trades:
                coin_wins = [t for t in coin_trades if t['result'] == 'WIN']
                per_coin[coin] = {
                    'trades': len(coin_trades),
                    'win_rate': round(len(coin_wins) / len(coin_trades) * 100, 1),
                    'total_pnl_usd': round(sum(t['pnl_usd'] for t in coin_trades), 2),
                    'avg_pnl_pct': round(np.mean([t['net_pnl_pct'] for t in coin_trades]), 2),
                }

        # Exit reason distribution
        from collections import Counter
        exit_reasons = dict(Counter(t['exit_reason'] for t in trades))

        return {
            'total_trades': total,
            'win_rate': round(win_rate, 1),
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'total_return_pct': total_return,
            'avg_win_pct': round(float(avg_win), 2),
            'avg_loss_pct': round(float(avg_loss), 2),
            'exit_reasons': exit_reasons,
            'per_coin': per_coin,
            'equity_curve': equity_curve,
        }


# Singleton instance
paper_trader = PaperTrader()
