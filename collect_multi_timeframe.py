import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import warnings
warnings.filterwarnings('ignore')

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
        
        # Moving averages
        for period in [7, 14, 21, 50]:
            features[f'{prefix}_sma_{period}'] = SMAIndicator(data['close'], period).sma_indicator()
            features[f'{prefix}_ema_{period}'] = EMAIndicator(data['close'], period).ema_indicator()
            features[f'{prefix}_dist_sma_{period}'] = (data['close'] - features[f'{prefix}_sma_{period}']) / data['close'] * 100
        
        # RSI
        features[f'{prefix}_rsi'] = RSIIndicator(data['close'], 14).rsi()
        
        # MACD
        macd = MACD(data['close'])
        features[f'{prefix}_macd'] = macd.macd()
        features[f'{prefix}_macd_signal'] = macd.macd_signal()
        features[f'{prefix}_macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(data['close'], 20, 2)
        features[f'{prefix}_bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        features[f'{prefix}_bb_position'] = (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # ATR
        features[f'{prefix}_atr'] = AverageTrueRange(data['high'], data['low'], data['close'], 14).average_true_range()
        features[f'{prefix}_atr_pct'] = features[f'{prefix}_atr'] / data['close'] * 100
        
        # ADX (trend strength)
        try:
            features[f'{prefix}_adx'] = ADXIndicator(data['high'], data['low'], data['close'], 14).adx()
        except:
            features[f'{prefix}_adx'] = 25  # default
        
        # Momentum
        features[f'{prefix}_momentum'] = data['close'] / data['close'].shift(10) - 1
        
        # Volume
        features[f'{prefix}_volume_ma'] = data['volume'].rolling(14).mean()
        features[f'{prefix}_volume_ratio'] = data['volume'] / features[f'{prefix}_volume_ma']
        
        # Add timestamp for merging
        features['timestamp'] = data['timestamp']
        
        return features
    
    # 1H features (base)
    print("  Adding 1H features...")
    df_1h_features = add_features_for_tf(dfs['1h'], '1h')
    df = df.merge(df_1h_features, on='timestamp', how='left')
    
    # 4H features (merge on nearest lower timestamp)
    if '4h' in dfs:
        print("  Adding 4H features...")
        df_4h_features = add_features_for_tf(dfs['4h'], '4h')
        df_4h_features['timestamp_4h'] = df_4h_features['timestamp']
        
        # Round 1h timestamp down to nearest 4h
        df['timestamp_4h'] = df['timestamp'].dt.floor('4h')
        df = df.merge(df_4h_features.drop(columns=['timestamp']), on='timestamp_4h', how='left')
        df = df.drop(columns=['timestamp_4h'])
    
    # 1D features
    if '1d' in dfs:
        print("  Adding 1D features...")
        df_1d_features = add_features_for_tf(dfs['1d'], '1d')
        df_1d_features['timestamp_1d'] = df_1d_features['timestamp']
        
        df['timestamp_1d'] = df['timestamp'].dt.floor('1d')
        df = df.merge(df_1d_features.drop(columns=['timestamp']), on='timestamp_1d', how='left')
        df = df.drop(columns=['timestamp_1d'])
    
    # 1W features
    if '1w' in dfs:
        print("  Adding 1W features...")
        df_1w_features = add_features_for_tf(dfs['1w'], '1w')
        df_1w_features['timestamp_1w'] = df_1w_features['timestamp']
        
        # Round to start of week
        df['timestamp_1w'] = df['timestamp'].dt.to_period('W').dt.start_time
        df_1w_features['timestamp_1w'] = df_1w_features['timestamp'].dt.to_period('W').dt.start_time
        df = df.merge(df_1w_features.drop(columns=['timestamp']), on='timestamp_1w', how='left')
        df = df.drop(columns=['timestamp_1w'])
    
    # Create target (24h forward return)
    df['target_return'] = (df['close'].shift(-24) / df['close'] - 1) * 100
    df['target_direction'] = df['target_return'].apply(
        lambda x: 'UP' if x > 3 else ('DOWN' if x < -3 else 'SIDEWAYS')
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
    
    coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'PEPE/USDT']
    
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