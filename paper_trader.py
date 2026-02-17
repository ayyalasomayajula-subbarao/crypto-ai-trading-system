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

from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "data", "paper_trading_state.json")
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# ── Frozen configuration (DO NOT CHANGE during paper trading) ──
CONFIG = {
    'coins': {
        'SOL_USDT': {'threshold': 0.35, 'model': 'wf_decision_model.pkl'},
        'PEPE_USDT': {'threshold': 0.40, 'model': 'wf_decision_model.pkl'},
    },
    'tp_pct': 0.05,
    'sl_pct': 0.03,
    'time_limit_hours': 48,
    'risk_per_trade': 0.005,   # 0.5% equity risk per trade
    'fee_pct': 0.0006,
    'slippage_pct': 0.0003,
    'spread_pct': 0.0002,
    'initial_capital': 10000,
    'check_interval_seconds': 60,  # check every 60s for new candle
}
CONFIG['round_trip_cost'] = (CONFIG['fee_pct'] + CONFIG['slippage_pct'] + CONFIG['spread_pct']) * 2


def add_features_for_tf(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute 27 technical features for one timeframe. Identical to collect_multi_timeframe.py."""
    features = pd.DataFrame(index=data.index)

    features[f'{prefix}_returns_1'] = data['close'].pct_change(1) * 100
    features[f'{prefix}_returns_5'] = data['close'].pct_change(5) * 100
    features[f'{prefix}_hl_range'] = (data['high'] - data['low']) / data['close'] * 100

    for period in [7, 14, 21, 50]:
        features[f'{prefix}_sma_{period}'] = SMAIndicator(data['close'], period).sma_indicator()
        features[f'{prefix}_ema_{period}'] = EMAIndicator(data['close'], period).ema_indicator()
        features[f'{prefix}_dist_sma_{period}'] = (data['close'] - features[f'{prefix}_sma_{period}']) / data['close'] * 100

    features[f'{prefix}_rsi'] = RSIIndicator(data['close'], 14).rsi()

    macd = MACD(data['close'])
    features[f'{prefix}_macd'] = macd.macd()
    features[f'{prefix}_macd_signal'] = macd.macd_signal()
    features[f'{prefix}_macd_diff'] = macd.macd_diff()

    bb = BollingerBands(data['close'], 20, 2)
    features[f'{prefix}_bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    features[f'{prefix}_bb_position'] = (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

    features[f'{prefix}_atr'] = AverageTrueRange(data['high'], data['low'], data['close'], 14).average_true_range()
    features[f'{prefix}_atr_pct'] = features[f'{prefix}_atr'] / data['close'] * 100

    try:
        features[f'{prefix}_adx'] = ADXIndicator(data['high'], data['low'], data['close'], 14).adx()
    except Exception:
        features[f'{prefix}_adx'] = 25.0

    features[f'{prefix}_momentum'] = data['close'] / data['close'].shift(10) - 1

    features[f'{prefix}_volume_ma'] = data['volume'].rolling(14).mean()
    features[f'{prefix}_volume_ratio'] = data['volume'] / features[f'{prefix}_volume_ma']

    features['timestamp'] = data['timestamp']
    return features


def fetch_klines(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    """Fetch OHLCV from Binance REST API (no auth needed)."""
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
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


def compute_live_features(coin: str) -> Optional[pd.Series]:
    """
    Fetch latest candles for all timeframes, compute 108 features,
    return the most recent row as a Series.
    """
    try:
        # Fetch all timeframes
        df_1h = fetch_klines(coin, '1h', 100)
        df_4h = fetch_klines(coin, '4h', 100)
        df_1d = fetch_klines(coin, '1d', 100)
        df_1w = fetch_klines(coin, '1w', 52)

        # Compute features per timeframe
        feat_1h = add_features_for_tf(df_1h, '1h')
        feat_4h = add_features_for_tf(df_4h, '4h')
        feat_1d = add_features_for_tf(df_1d, '1d')
        feat_1w = add_features_for_tf(df_1w, '1w')

        # Merge onto 1h base (same logic as collect_multi_timeframe.py)
        df = df_1h.copy()
        df = df.merge(feat_1h, on='timestamp', how='left')

        # 4H: floor to nearest 4h
        feat_4h['timestamp_4h'] = feat_4h['timestamp']
        df['timestamp_4h'] = df['timestamp'].dt.floor('4h')
        df = df.merge(feat_4h.drop(columns=['timestamp']), on='timestamp_4h', how='left')
        df.drop(columns=['timestamp_4h'], inplace=True)

        # 1D: floor to nearest day
        feat_1d['timestamp_1d'] = feat_1d['timestamp']
        df['timestamp_1d'] = df['timestamp'].dt.floor('1d')
        df = df.merge(feat_1d.drop(columns=['timestamp']), on='timestamp_1d', how='left')
        df.drop(columns=['timestamp_1d'], inplace=True)

        # 1W: round to start of week
        feat_1w['timestamp_1w'] = feat_1w['timestamp']
        df['timestamp_1w'] = df['timestamp'].dt.to_period('W').dt.start_time
        feat_1w['timestamp_1w'] = feat_1w['timestamp'].dt.to_period('W').dt.start_time
        df = df.merge(feat_1w.drop(columns=['timestamp']), on='timestamp_1w', how='left')
        df.drop(columns=['timestamp_1w'], inplace=True)

        # Drop NaN rows and return last
        df = df.dropna()
        if len(df) == 0:
            return None

        return df.iloc[-1]

    except Exception as e:
        print(f"  [PAPER] Feature computation error for {coin}: {e}")
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
        """Load frozen walk-forward models."""
        for coin, cfg in CONFIG['coins'].items():
            model_path = os.path.join(BASE_DIR, f"models/{coin}/{cfg['model']}")
            features_path = os.path.join(BASE_DIR, f"models/{coin}/decision_features.txt")

            if not os.path.exists(model_path):
                print(f"  [PAPER] WARNING: Model not found: {model_path}")
                continue

            model = joblib.load(model_path)
            # Patch sklearn version mismatch
            if not hasattr(model, 'monotonic_cst'):
                model.monotonic_cst = None
            if hasattr(model, 'estimators_'):
                for est in model.estimators_:
                    if not hasattr(est, 'monotonic_cst'):
                        est.monotonic_cst = None

            with open(features_path) as f:
                feature_cols = [line.strip() for line in f if line.strip()]

            self.models[coin] = {
                'model': model,
                'classes': list(model.classes_),
                'win_idx': list(model.classes_).index('WIN'),
                'loss_idx': list(model.classes_).index('LOSS'),
            }
            self.feature_cols[coin] = feature_cols
            print(f"  [PAPER] Loaded model: {coin} ({cfg['model']})")

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
                    'tp_pct': CONFIG['tp_pct'],
                    'sl_pct': CONFIG['sl_pct'],
                    'time_limit_hours': CONFIG['time_limit_hours'],
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
        """Check if model generates entry signal."""
        features = compute_live_features(coin)
        if features is None:
            return

        model_info = self.models[coin]
        feature_cols = self.feature_cols[coin]
        threshold = CONFIG['coins'][coin]['threshold']

        # Extract features in model order
        try:
            X = pd.DataFrame([[features.get(col, 0) for col in feature_cols]], columns=feature_cols)
            X = X.fillna(0)
        except Exception as e:
            print(f"  [PAPER] Feature extraction error {coin}: {e}")
            return

        # Predict
        probas = model_info['model'].predict_proba(X)[0]
        # Normalize (sklearn version mismatch safety)
        total = probas.sum()
        if total > 0:
            probas = probas / total

        win_prob = float(probas[model_info['win_idx']])
        loss_prob = float(probas[model_info['loss_idx']])

        # Entry signal check (same as backtest)
        if win_prob >= threshold and loss_prob < 0.40:
            # Get next candle's open for entry (use current price as approximation)
            # In live trading, we place order and get filled at next open
            entry_price = float(features.get('close', 0))
            if entry_price <= 0:
                return

            # Position size: 0.5% equity risk
            equity = self.state['equity']
            risk_amount = equity * CONFIG['risk_per_trade']
            # Size based on SL distance
            sl_distance = CONFIG['sl_pct']
            position_size_usd = risk_amount / sl_distance

            tp_price = entry_price * (1 + CONFIG['tp_pct'])
            sl_price = entry_price * (1 - CONFIG['sl_pct'])

            position = {
                'coin': coin,
                'entry_time': now_str,
                'entry_price': entry_price,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'position_size_usd': round(position_size_usd, 2),
                'win_prob': round(win_prob * 100, 1),
                'loss_prob': round(loss_prob * 100, 1),
                'candles_held': 0,
            }
            self.state['positions'][coin] = position
            print(f"  [PAPER] ENTRY {coin}: ${entry_price:.4f} | "
                  f"TP=${tp_price:.4f} SL=${sl_price:.4f} | "
                  f"Size=${position_size_usd:.2f} | Win={win_prob*100:.1f}%")

    def _check_exit(self, coin: str, candle: pd.Series, now_str: str):
        """Check if open position should be closed."""
        pos = self.state['positions'][coin]
        pos['candles_held'] += 1

        high = float(candle['high'])
        low = float(candle['low'])
        close = float(candle['close'])

        entry_price = pos['entry_price']
        tp_price = pos['tp_price']
        sl_price = pos['sl_price']

        exit_reason = None
        exit_price = None

        tp_hit = high >= tp_price
        sl_hit = low <= sl_price

        # Worst-case rule: both hit → SL wins
        if tp_hit and sl_hit:
            exit_reason = 'SL'
            exit_price = sl_price
        elif sl_hit:
            exit_reason = 'SL'
            exit_price = sl_price
        elif tp_hit:
            exit_reason = 'TP'
            exit_price = tp_price
        elif pos['candles_held'] >= CONFIG['time_limit_hours']:
            exit_reason = 'TIME'
            exit_price = close

        if exit_reason:
            gross_pnl_pct = (exit_price - entry_price) / entry_price * 100
            net_pnl_pct = gross_pnl_pct - (CONFIG['round_trip_cost'] * 100)
            pnl_usd = pos['position_size_usd'] * (net_pnl_pct / 100)

            self.state['equity'] += pnl_usd

            trade = {
                'coin': coin,
                'entry_time': pos['entry_time'],
                'exit_time': now_str,
                'entry_price': round(entry_price, 8),
                'exit_price': round(exit_price, 8),
                'position_size_usd': pos['position_size_usd'],
                'win_prob': pos['win_prob'],
                'loss_prob': pos['loss_prob'],
                'gross_pnl_pct': round(gross_pnl_pct, 4),
                'net_pnl_pct': round(net_pnl_pct, 4),
                'pnl_usd': round(pnl_usd, 2),
                'exit_reason': exit_reason,
                'hours_held': pos['candles_held'],
                'result': 'WIN' if net_pnl_pct > 0 else 'LOSS',
                'equity_after': round(self.state['equity'], 2),
            }
            self.state['trades'].append(trade)
            del self.state['positions'][coin]

            print(f"  [PAPER] EXIT {coin}: {exit_reason} | "
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
