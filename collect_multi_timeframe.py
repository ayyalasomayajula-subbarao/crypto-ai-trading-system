import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Simple direction thresholds — % 24h return to qualify as UP / DOWN.
# Coin-specific, calibrated so UP and DOWN are each ~20-35% of bars.
DIRECTION_THRESHOLDS = {
    'BTC_USDT':  1.5,
    'ETH_USDT':  1.5,
    'SOL_USDT':  3.0,
    'PEPE_USDT': 5.0,
    'AVAX_USDT': 2.5,
    'BNB_USDT':  2.0,
    'LINK_USDT': 2.5,
    'ARB_USDT':  3.0,
    'OP_USDT':   3.0,
    'INJ_USDT':  3.5,
}

# Triple Barrier parameters — preserved for research; not used in production labeling.
# The current feature set (ADX, SMA, multi-TF technicals) predicts direction reliably
# but cannot predict intrabar TP/SL path dynamics well enough to use TB labels.
TRIPLE_BARRIER_PARAMS = {
    'BTC_USDT':  {'tp': 0.030, 'sl': 0.015, 'time_limit': 48},
    'ETH_USDT':  {'tp': 0.045, 'sl': 0.015, 'time_limit': 48},
    'SOL_USDT':  {'tp': 0.075, 'sl': 0.025, 'time_limit': 72},
    'PEPE_USDT': {'tp': 0.150, 'sl': 0.050, 'time_limit': 48},
    'AVAX_USDT': {'tp': 0.075, 'sl': 0.025, 'time_limit': 72},
    'BNB_USDT':  {'tp': 0.060, 'sl': 0.020, 'time_limit': 48},
    'LINK_USDT': {'tp': 0.075, 'sl': 0.025, 'time_limit': 72},
    'ARB_USDT':  {'tp': 0.075, 'sl': 0.025, 'time_limit': 72},
    'OP_USDT':   {'tp': 0.075, 'sl': 0.025, 'time_limit': 72},
    'INJ_USDT':  {'tp': 0.090, 'sl': 0.030, 'time_limit': 72},
}


def compute_triple_barrier_labels(df: pd.DataFrame, tp_pct: float,
                                   sl_pct: float, time_limit: int) -> list:
    """
    Bidirectional Triple Barrier labeling over a 1H OHLCV DataFrame.

    For each bar i, evaluate BOTH directions independently:
      UP       — LONG TP  (entry*(1+tp)) hit before LONG SL  (entry*(1-sl))
                 → a long trade from this entry would profit
      DOWN     — SHORT TP (entry*(1-tp)) hit before SHORT SL (entry*(1+sl))
                 → a short trade from this entry would profit
      SIDEWAYS — neither LONG nor SHORT TP reached within time_limit bars

    When both LONG and SHORT would profit (price whipsaws), label by whichever
    TP fires first.  This is the correct bidirectional formulation: DOWN means
    "short would win", NOT "long would lose" (those are different events).

    Note: with asymmetric TP/SL the expected random-walk frequencies are:
      P(UP) = P(DOWN) ≈ sl/(tp+sl)  (e.g. 33% for 3%/1.5% BTC params)
    giving a reasonably balanced label set.
    """
    n      = len(df)
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    labels = ['SIDEWAYS'] * n

    for i in range(n - 1):
        entry     = closes[i]
        long_tp   = entry * (1 + tp_pct)   # LONG profit target
        long_sl   = entry * (1 - sl_pct)   # LONG stop loss
        short_tp  = entry * (1 - tp_pct)   # SHORT profit target
        short_sl  = entry * (1 + sl_pct)   # SHORT stop loss

        end          = min(i + time_limit + 1, n)
        future_highs = highs[i + 1 : end]
        future_lows  = lows[i + 1 : end]

        long_tp_hits  = future_highs >= long_tp
        long_sl_hits  = future_lows  <= long_sl
        short_tp_hits = future_lows  <= short_tp
        short_sl_hits = future_highs >= short_sl

        long_tp_bar  = int(np.argmax(long_tp_hits))  if long_tp_hits.any()  else time_limit + 1
        long_sl_bar  = int(np.argmax(long_sl_hits))  if long_sl_hits.any()  else time_limit + 1
        short_tp_bar = int(np.argmax(short_tp_hits)) if short_tp_hits.any() else time_limit + 1
        short_sl_bar = int(np.argmax(short_sl_hits)) if short_sl_hits.any() else time_limit + 1

        long_wins  = long_tp_hits.any()  and long_tp_bar  <= long_sl_bar
        short_wins = short_tp_hits.any() and short_tp_bar <= short_sl_bar

        if long_wins and short_wins:
            # Both would profit (whipsaw) — take whichever TP fires first
            labels[i] = 'UP' if long_tp_bar <= short_tp_bar else 'DOWN'
        elif long_wins:
            labels[i] = 'UP'
        elif short_wins:
            labels[i] = 'DOWN'
        # else: SIDEWAYS (neither TP reached in time)

    return labels

class MultiTimeframeCollector:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.timeframes = ['1h', '4h', '1d', '1w']
    
    def fetch_ohlcv(self, symbol, timeframe, days=None):
        """Fetch OHLCV data"""
        print(f"  Fetching {symbol} {timeframe}...")
        
        # Start from 2017 for max history
        since = self.exchange.parse8601('2017-01-01T00:00:00Z')
        
        all_ohlcv = []
        
        while since < self.exchange.milliseconds():
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"    Error: {e}")
                time.sleep(5)
                continue
        
        if not all_ohlcv:
            return None
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        print(f"    ✅ {len(df):,} candles ({df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')})")
        
        return df
    
    def collect_all_timeframes(self, symbol):
        """Collect all timeframes for a symbol"""
        coin = symbol.replace('/', '_')
        data = {}
        
        print(f"\n{'='*50}")
        print(f"Collecting {symbol}")
        print(f"{'='*50}")
        
        for tf in self.timeframes:
            df = self.fetch_ohlcv(symbol, tf)
            
            if df is not None:
                # Save individual file
                filename = f"data/{coin}_{tf}.csv"
                df.to_csv(filename, index=False)
                data[tf] = df
        
        return data


def create_multi_timeframe_features(coin):
    """Create aligned multi-timeframe features for a coin"""
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import BollingerBands, AverageTrueRange
    import numpy as np
    
    print(f"\n{'='*50}")
    print(f"Creating Multi-TF Features: {coin}")
    print(f"{'='*50}")
    
    # Load all timeframes
    dfs = {}
    for tf in ['1h', '4h', '1d', '1w']:
        filepath = f"data/{coin}_{tf}.csv"
        if os.path.exists(filepath):
            dfs[tf] = pd.read_csv(filepath)
            dfs[tf]['timestamp'] = pd.to_datetime(dfs[tf]['timestamp'])
            print(f"  Loaded {tf}: {len(dfs[tf]):,} rows")
    
    if '1h' not in dfs:
        print(f"  ❌ Missing 1h data for {coin}")
        return None
    
    # Base: 1H data
    df = dfs['1h'].copy()
    
    def add_features_for_tf(data, prefix):
        """Add technical features with prefix"""
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features[f'{prefix}_returns_1'] = data['close'].pct_change(1) * 100
        features[f'{prefix}_returns_5'] = data['close'].pct_change(5) * 100
        features[f'{prefix}_hl_range'] = (data['high'] - data['low']) / data['close'] * 100
        
        # Moving averages — sma_21 and sma_50 only.
        # Removed: sma_7/14 (too noisy), all EMAs (redundant with SMA, lowest importance).
        for period in [21, 50]:
            features[f'{prefix}_sma_{period}'] = SMAIndicator(data['close'], period).sma_indicator()
            features[f'{prefix}_dist_sma_{period}'] = (data['close'] - features[f'{prefix}_sma_{period}']) / data['close'] * 100

        # RSI
        features[f'{prefix}_rsi'] = RSIIndicator(data['close'], 14).rsi()

        # MACD — keep only macd_diff (histogram); macd/macd_signal are redundant.
        macd_ind = MACD(data['close'])
        features[f'{prefix}_macd_diff'] = macd_ind.macd_diff()

        # Bollinger Bands
        bb = BollingerBands(data['close'], 20, 2)
        features[f'{prefix}_bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        features[f'{prefix}_bb_position'] = (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

        # ATR % only — raw ATR is price-scale-dependent and redundant with atr_pct.
        _atr = AverageTrueRange(data['high'], data['low'], data['close'], 14).average_true_range()
        features[f'{prefix}_atr_pct'] = _atr / data['close'] * 100

        # ADX (trend strength)
        try:
            features[f'{prefix}_adx'] = ADXIndicator(data['high'], data['low'], data['close'], 14).adx()
        except:
            features[f'{prefix}_adx'] = 25  # default

        # Momentum
        features[f'{prefix}_momentum'] = data['close'] / data['close'].shift(10) - 1

        # SMA slope — trend acceleration: how fast is the SMA rising/falling?
        # Lookback ≈ 24h worth of bars per timeframe.
        _slope_bars = {'1h': 24, '4h': 6, '1d': 5, '1w': 4, '15m': 96}.get(prefix, 24)
        for _p in [21, 50]:
            _sma = features[f'{prefix}_sma_{_p}']
            features[f'{prefix}_sma{_p}_slope'] = (
                (_sma - _sma.shift(_slope_bars)) / _sma.shift(_slope_bars).abs().clip(lower=1e-8) * 100
            )

        # Realized volatility — volatility regime detection.
        # 1h bars: rolling std of returns over 24h and 7d windows.
        # Only computed for 1h (finest granularity available); coarser TFs don't add info.
        if prefix == '1h':
            _ret = data['close'].pct_change()
            features[f'{prefix}_realized_vol_24h'] = _ret.rolling(24,  min_periods=12).std() * 100
            features[f'{prefix}_realized_vol_7d']  = _ret.rolling(168, min_periods=48).std() * 100
            features[f'{prefix}_vol_regime_ratio'] = (
                features[f'{prefix}_realized_vol_24h'] /
                features[f'{prefix}_realized_vol_7d'].clip(lower=1e-4)
            )  # >1 = vol expanding (breakout/crash), <1 = vol contracting (consolidation)

        # Pullback depth from 10-period high — captures entry quality within trend.
        # 0% = price AT the 10-period high (stretched, poor long entry).
        # -5% = 5% below 10-period high (healthy pullback, better entry).
        features[f'{prefix}_dist_from_10p_high'] = (
            (data['close'] - data['close'].rolling(10).max()) / data['close'].rolling(10).max() * 100
        )

        # Volume ratio — relative volume vs 14-period MA.
        # Removed: volume_ma absolute (price-scale-dependent, low importance).
        _vol_ma = data['volume'].rolling(14).mean()
        features[f'{prefix}_volume_ratio'] = data['volume'] / _vol_ma.clip(lower=1e-8)

        # Add timestamp for merging
        features['timestamp'] = data['timestamp']
        
        return features
    
    # 15m timeframe removed — adding 28 highly-correlated features caused regression
    # (LINK dropped MARGINAL→NOT_VIABLE; meta model over-traded, WR fell below 44%).
    # 15m data is still collected (useful for future research) but not used in ML features.

    # 1H features (base)
    print("  Adding 1H features...")
    df_1h_features = add_features_for_tf(dfs['1h'], '1h')
    df = df.merge(df_1h_features, on='timestamp', how='left')
    
    # 4H features — use the PREVIOUS completed 4h candle to avoid lookahead.
    # At 1h bar T, floor('4h') gives the START of the current (incomplete) 4h candle.
    # Subtracting 4h gives the last FULLY CLOSED 4h candle.
    if '4h' in dfs:
        print("  Adding 4H features...")
        df_4h_features = add_features_for_tf(dfs['4h'], '4h')
        df_4h_features['timestamp_4h'] = df_4h_features['timestamp']

        df['timestamp_4h'] = df['timestamp'].dt.floor('4h') - pd.Timedelta(hours=4)
        df = df.merge(df_4h_features.drop(columns=['timestamp']), on='timestamp_4h', how='left')
        df = df.drop(columns=['timestamp_4h'])

    # 1D features — use the PREVIOUS completed daily candle to avoid lookahead.
    # At any hour of day D, floor('1d') gives midnight of D.
    # The day-D candle closes at midnight of D+1, so we subtract 1 day to get
    # the fully completed day-(D-1) candle.
    if '1d' in dfs:
        print("  Adding 1D features...")
        df_1d_features = add_features_for_tf(dfs['1d'], '1d')
        df_1d_features['timestamp_1d'] = df_1d_features['timestamp']

        df['timestamp_1d'] = df['timestamp'].dt.floor('1d') - pd.Timedelta(days=1)
        df = df.merge(df_1d_features.drop(columns=['timestamp']), on='timestamp_1d', how='left')
        df = df.drop(columns=['timestamp_1d'])

    # 1W features — use the PREVIOUS completed weekly candle to avoid lookahead.
    if '1w' in dfs:
        print("  Adding 1W features...")
        df_1w_features = add_features_for_tf(dfs['1w'], '1w')
        df_1w_features['timestamp_1w'] = df_1w_features['timestamp']

        # Current week's start, then step back 1 week to get last completed week
        df['timestamp_1w'] = (
            df['timestamp'].dt.to_period('W').dt.start_time - pd.Timedelta(weeks=1)
        )
        df_1w_features['timestamp_1w'] = df_1w_features['timestamp'].dt.to_period('W').dt.start_time
        df = df.merge(df_1w_features.drop(columns=['timestamp']), on='timestamp_1w', how='left')
        df = df.drop(columns=['timestamp_1w'])

    # Funding rate features — perpetual futures sentiment signal.
    # Use merge_asof (backward) so each 1h bar gets the MOST RECENTLY SETTLED
    # 8h funding rate. No lookahead: the settled rate is published at settlement time.
    funding_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/{coin}_funding.csv')
    if os.path.exists(funding_path):
        print("  Adding funding rate features...")
        df_fund = pd.read_csv(funding_path)
        df_fund['timestamp'] = pd.to_datetime(df_fund['timestamp'])
        df_fund = df_fund.sort_values('timestamp')

        # Rolling stats on the funding rate series
        df_fund['funding_rate_3d_avg']  = df_fund['funding_rate'].rolling(9,  min_periods=1).mean()   # 9×8h  = 3 days
        df_fund['funding_rate_7d_avg']  = df_fund['funding_rate'].rolling(21, min_periods=1).mean()  # 21×8h = 7 days
        df_fund['funding_rate_momentum'] = df_fund['funding_rate'].diff(3)  # change over last 3 periods (24h)
        # Funding trend: rate of acceleration — is funding drifting toward extreme?
        # diff(3)/3 = avg change per 8h period over last 3 periods (normalized slope).
        df_fund['funding_trend'] = df_fund['funding_rate'].diff(3) / 3

        df = pd.merge_asof(
            df.sort_values('timestamp'),
            df_fund[['timestamp', 'funding_rate', 'funding_rate_3d_avg',
                      'funding_rate_7d_avg', 'funding_rate_momentum', 'funding_trend']],
            on='timestamp',
            direction='backward',
        )
    else:
        print(f"  ⚠️  No funding rate file for {coin} — skipping")

    # Taker buy/sell volume — aggressive order flow (institutional intent proxy).
    # Source: Binance klines col 9 (taker_buy_base_asset_volume).
    # No lookahead: taker volume is settled at candle close, same as OHLCV.
    taker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/{coin}_taker_volume.csv')
    if os.path.exists(taker_path):
        print("  Adding taker volume features...")
        df_tv = pd.read_csv(taker_path, parse_dates=['timestamp'])
        df_tv = df_tv.sort_values('timestamp')

        # Signed imbalance: +1 = pure buy pressure, -1 = pure sell pressure
        _vol   = df_tv['volume'].clip(lower=1e-8)
        _imb   = (df_tv['taker_buy_volume'] - df_tv['taker_sell_volume']) / _vol
        # Rolling MAs only — raw imbalance and slope have lowest importance; keep smoothed.
        df_tv['imbalance_4h_ma']   = _imb.rolling(4,  min_periods=2).mean()
        df_tv['imbalance_24h_ma']  = _imb.rolling(24, min_periods=8).mean()

        df = pd.merge_asof(
            df.sort_values('timestamp'),
            df_tv[['timestamp', 'imbalance_4h_ma', 'imbalance_24h_ma']],
            on='timestamp',
            direction='backward',
        )
    else:
        print(f"  ⚠️  No taker volume file for {coin} — run collect_taker_volume.py first")

    # Simple 24h forward return threshold labels.
    # Empirically outperforms Triple Barrier because the feature set (ADX, SMA, etc.)
    # predicts direction well but cannot reliably predict intrabar TP/SL path dynamics.
    # Triple Barrier is preserved in compute_triple_barrier_labels() for future research.
    df['target_return'] = (df['close'].shift(-24) / df['close'] - 1) * 100
    thresh = DIRECTION_THRESHOLDS.get(coin, 3.0)
    df['target_direction'] = df['target_return'].apply(
        lambda x: 'UP' if x > thresh else ('DOWN' if x < -thresh else 'SIDEWAYS')
    )
    
    # Drop NaN
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    
    print(f"\n  Total features: {len(df.columns)}")
    print(f"  Rows: {rows_after:,} (dropped {rows_before - rows_after:,})")
    
    # Target distribution
    dist = df['target_direction'].value_counts(normalize=True) * 100
    print(f"\n  Target distribution:")
    for label in ['UP', 'SIDEWAYS', 'DOWN']:
        if label in dist.index:
            print(f"    {label}: {dist[label]:.1f}%")
    
    # Save
    output_file = f"data/{coin}_multi_tf_features.csv"
    df.to_csv(output_file, index=False)
    print(f"\n  ✅ Saved: {output_file}")
    
    return df


# Main
if __name__ == "__main__":
    print("\n" + "#"*60)
    print("  MULTI-TIMEFRAME DATA COLLECTION")
    print("#"*60)
    
    collector = MultiTimeframeCollector()
    
    coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'PEPE/USDT',
             'AVAX/USDT', 'BNB/USDT', 'LINK/USDT',
             'ARB/USDT', 'OP/USDT', 'INJ/USDT']
    
    # Step 1: Collect all timeframes
    print("\n" + "="*60)
    print("STEP 1: COLLECTING DATA")
    print("="*60)
    
    for coin in coins:
        collector.collect_all_timeframes(coin)
    
    # Step 2: Create multi-TF features
    print("\n" + "="*60)
    print("STEP 2: CREATING MULTI-TIMEFRAME FEATURES")
    print("="*60)
    
    for coin in coins:
        coin_name = coin.replace('/', '_')
        create_multi_timeframe_features(coin_name)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for f in sorted(os.listdir('data')):
        if f.endswith('_multi_tf_features.csv'):
            filepath = f"data/{f}"
            df = pd.read_csv(filepath)
            print(f"\n  {f}:")
            print(f"    Rows: {len(df):,}")
            print(f"    Features: {len(df.columns)}")
    
    print("\n" + "="*60)
    print("🎉 MULTI-TIMEFRAME DATA READY!")
    print("="*60)