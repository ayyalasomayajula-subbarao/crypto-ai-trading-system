import pandas as pd
import numpy as np
import joblib
import ccxt
from datetime import datetime, timedelta
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FEATURE ENGINEERING (Simplified for real-time)
# ============================================================

class RealtimeFeatureEngineer:
    """Create features from live OHLCV data"""
    
    def __init__(self):
        from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
        from ta.momentum import RSIIndicator, StochasticOscillator
        from ta.volatility import BollingerBands, AverageTrueRange
        
        self.SMAIndicator = SMAIndicator
        self.EMAIndicator = EMAIndicator
        self.MACD = MACD
        self.ADXIndicator = ADXIndicator
        self.RSIIndicator = RSIIndicator
        self.StochasticOscillator = StochasticOscillator
        self.BollingerBands = BollingerBands
        self.AverageTrueRange = AverageTrueRange
    
    def add_features(self, df, prefix):
        """Add technical features with prefix"""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features[f'{prefix}_returns_1'] = df['close'].pct_change(1) * 100
        features[f'{prefix}_returns_5'] = df['close'].pct_change(5) * 100
        features[f'{prefix}_hl_range'] = (df['high'] - df['low']) / df['close'] * 100
        
        # Moving averages
        for period in [7, 14, 21, 50]:
            features[f'{prefix}_sma_{period}'] = self.SMAIndicator(df['close'], period).sma_indicator()
            features[f'{prefix}_ema_{period}'] = self.EMAIndicator(df['close'], period).ema_indicator()
            features[f'{prefix}_dist_sma_{period}'] = (df['close'] - features[f'{prefix}_sma_{period}']) / df['close'] * 100
        
        # RSI
        features[f'{prefix}_rsi'] = self.RSIIndicator(df['close'], 14).rsi()
        
        # MACD
        macd = self.MACD(df['close'])
        features[f'{prefix}_macd'] = macd.macd()
        features[f'{prefix}_macd_signal'] = macd.macd_signal()
        features[f'{prefix}_macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = self.BollingerBands(df['close'], 20, 2)
        features[f'{prefix}_bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        features[f'{prefix}_bb_position'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # ATR
        features[f'{prefix}_atr'] = self.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
        features[f'{prefix}_atr_pct'] = features[f'{prefix}_atr'] / df['close'] * 100
        
        # ADX
        try:
            features[f'{prefix}_adx'] = self.ADXIndicator(df['high'], df['low'], df['close'], 14).adx()
        except:
            features[f'{prefix}_adx'] = 25
        
        # Momentum
        features[f'{prefix}_momentum'] = df['close'] / df['close'].shift(10) - 1
        
        # Volume
        features[f'{prefix}_volume_ma'] = df['volume'].rolling(14).mean()
        features[f'{prefix}_volume_ratio'] = df['volume'] / features[f'{prefix}_volume_ma']
        
        return features


# ============================================================
# LIVE DATA FETCHER
# ============================================================

class LiveDataFetcher:
    """Fetch live data from Binance"""
    
    def __init__(self):
        self.exchange = ccxt.binance()
        self.feature_engineer = RealtimeFeatureEngineer()
    
    def fetch_ohlcv(self, symbol, timeframe, limit=500):
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def fetch_multi_timeframe(self, symbol):
        """Fetch all timeframes and create features"""
        
        # Fetch each timeframe
        df_1h = self.fetch_ohlcv(symbol, '1h', 500)
        df_4h = self.fetch_ohlcv(symbol, '4h', 200)
        df_1d = self.fetch_ohlcv(symbol, '1d', 100)
        df_1w = self.fetch_ohlcv(symbol, '1w', 50)
        
        if df_1h is None:
            return None
        
        # Create features for each timeframe
        features_1h = self.feature_engineer.add_features(df_1h, '1h')
        features_1h['timestamp'] = df_1h['timestamp']
        features_1h['close'] = df_1h['close']
        
        # Merge 4H features
        if df_4h is not None:
            features_4h = self.feature_engineer.add_features(df_4h, '4h')
            features_4h['timestamp_4h'] = df_4h['timestamp']
            
            features_1h['timestamp_4h'] = features_1h['timestamp'].dt.floor('4h')
            features_1h = features_1h.merge(
                features_4h.assign(timestamp_4h=df_4h['timestamp']),
                on='timestamp_4h',
                how='left'
            )
            features_1h = features_1h.drop(columns=['timestamp_4h'])
        
        # Merge 1D features
        if df_1d is not None:
            features_1d = self.feature_engineer.add_features(df_1d, '1d')
            features_1d['timestamp_1d'] = df_1d['timestamp']
            
            features_1h['timestamp_1d'] = features_1h['timestamp'].dt.floor('1d')
            features_1h = features_1h.merge(
                features_1d.assign(timestamp_1d=df_1d['timestamp']),
                on='timestamp_1d',
                how='left'
            )
            features_1h = features_1h.drop(columns=['timestamp_1d'])
        
        # Merge 1W features
        if df_1w is not None:
            features_1w = self.feature_engineer.add_features(df_1w, '1w')
            features_1w['timestamp_1w'] = df_1w['timestamp']
            
            features_1h['timestamp_1w'] = features_1h['timestamp'].dt.to_period('W').dt.start_time
            features_1w['timestamp_1w'] = df_1w['timestamp'].dt.to_period('W').dt.start_time
            features_1h = features_1h.merge(
                features_1w,
                on='timestamp_1w',
                how='left'
            )
            features_1h = features_1h.drop(columns=['timestamp_1w'])
        
        return features_1h


# ============================================================
# SIGNAL SCANNER
# ============================================================

class SignalScanner:
    """Scan for trading signals across all coins"""
    
    def __init__(self):
        self.coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'PEPE/USDT']
        self.fetcher = LiveDataFetcher()
        self.models = {}
        self.feature_cols = {}
        
        # Thresholds
        self.thresholds = {
            'BTC_USDT': {'buy': 0.45, 'strong_buy': 0.50},
            'ETH_USDT': {'buy': 0.45, 'strong_buy': 0.50},
            'SOL_USDT': {'buy': 0.45, 'strong_buy': 0.50},
            'PEPE_USDT': {'buy': 0.50, 'strong_buy': 0.55}
        }
        
        self.load_models()
    
    def load_models(self):
        """Load decision models for all coins"""
        print("Loading models...")
        
        for coin in self.coins:
            coin_id = coin.replace('/', '_')
            model_path = f"models/{coin_id}/decision_model.pkl"
            features_path = f"models/{coin_id}/decision_features.txt"
            
            if os.path.exists(model_path):
                self.models[coin_id] = joblib.load(model_path)
                
                with open(features_path, 'r') as f:
                    self.feature_cols[coin_id] = [line.strip() for line in f.readlines()]
                
                print(f"  ✅ {coin_id} model loaded")
            else:
                print(f"  ❌ {coin_id} model not found")
    
    def get_signal(self, coin):
        """Get signal for a single coin"""
        coin_id = coin.replace('/', '_')
        
        if coin_id not in self.models:
            return None
        
        # Fetch live data
        features = self.fetcher.fetch_multi_timeframe(coin)
        
        if features is None or len(features) == 0:
            return None
        
        # Get latest row
        latest = features.tail(1)
        
        # Get required features
        required_features = self.feature_cols[coin_id]
        
        # Check for missing features
        available = [f for f in required_features if f in latest.columns]
        missing = [f for f in required_features if f not in latest.columns]
        
        if len(missing) > 0:
            # Fill missing with 0 (not ideal but allows prediction)
            for m in missing:
                latest[m] = 0
        
        X = latest[required_features]
        
        # Handle NaN
        X = X.fillna(0)
        
        # Get prediction
        model = self.models[coin_id]
        probas = model.predict_proba(X)[0]
        classes = list(model.classes_)
        
        win_idx = classes.index('WIN') if 'WIN' in classes else 0
        loss_idx = classes.index('LOSS') if 'LOSS' in classes else 1
        
        win_prob = probas[win_idx]
        loss_prob = probas[loss_idx]
        
        # Get current price
        current_price = latest['close'].values[0]
        
        # Determine signal
        thresholds = self.thresholds[coin_id]
        
        if win_prob >= thresholds['strong_buy']:
            signal = 'STRONG BUY 🟢🟢'
            confidence = 'HIGH'
        elif win_prob >= thresholds['buy']:
            signal = 'BUY 🟢'
            confidence = 'MEDIUM'
        elif loss_prob >= 0.50:
            signal = 'AVOID 🔴'
            confidence = 'HIGH'
        elif loss_prob >= 0.40:
            signal = 'CAUTION ⚠️'
            confidence = 'MEDIUM'
        else:
            signal = 'WAIT ⏳'
            confidence = 'LOW'
        
        return {
            'coin': coin,
            'price': current_price,
            'win_prob': win_prob,
            'loss_prob': loss_prob,
            'signal': signal,
            'confidence': confidence,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def scan_all(self):
        """Scan all coins for signals"""
        results = []
        
        for coin in self.coins:
            result = self.get_signal(coin)
            if result:
                results.append(result)
            time.sleep(1)  # Rate limiting
        
        return results
    
    def display_results(self, results):
        """Display scan results"""
        print("\n" + "="*80)
        print(f"  📡 SIGNAL SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Sort by WIN probability
        results = sorted(results, key=lambda x: x['win_prob'], reverse=True)
        
        print(f"\n  {'COIN':<12} {'PRICE':<15} {'WIN %':<10} {'LOSS %':<10} {'SIGNAL':<18} {'CONF':<8}")
        print("  " + "-"*75)
        
        buy_signals = []
        
        for r in results:
            price_str = f"${r['price']:.2f}" if r['price'] > 1 else f"${r['price']:.8f}"
            
            print(f"  {r['coin']:<12} {price_str:<15} {r['win_prob']*100:>5.1f}%    {r['loss_prob']*100:>5.1f}%    {r['signal']:<18} {r['confidence']:<8}")
            
            if 'BUY' in r['signal']:
                buy_signals.append(r)
        
        print("  " + "-"*75)
        
        # Summary
        if buy_signals:
            print(f"\n  🚨 BUY SIGNALS DETECTED!")
            for b in buy_signals:
                print(f"     → {b['coin']}: {b['signal']} (WIN: {b['win_prob']*100:.1f}%)")
        else:
            print(f"\n  ⏳ No buy signals right now. Market conditions unfavorable.")
        
        return buy_signals


def run_scanner_once():
    """Run scanner once"""
    scanner = SignalScanner()
    results = scanner.scan_all()
    buy_signals = scanner.display_results(results)
    return results, buy_signals


def run_continuous_scanner(interval_minutes=15):
    """Run scanner continuously"""
    print("\n" + "#"*80)
    print("  🔄 CONTINUOUS SIGNAL SCANNER")
    print(f"  Scanning every {interval_minutes} minutes. Press Ctrl+C to stop.")
    print("#"*80)
    
    scanner = SignalScanner()
    
    while True:
        try:
            results = scanner.scan_all()
            buy_signals = scanner.display_results(results)
            
            if buy_signals:
                print("\n  🔔 ALERT: Buy signal detected! Check above for details.")
            
            # Wait for next scan
            print(f"\n  ⏰ Next scan in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            print("\n\n  Scanner stopped by user.")
            break
        except Exception as e:
            print(f"\n  Error: {e}")
            print("  Retrying in 1 minute...")
            time.sleep(60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "#"*80)
    print("  📡 CRYPTO SIGNAL SCANNER")
    print("#"*80)
    
    print("\n  Options:")
    print("  1. Run once (single scan)")
    print("  2. Run continuously (scan every 15 min)")
    print("  3. Run continuously (scan every 1 hour)")
    
    choice = input("\n  Enter choice (1/2/3): ").strip()
    
    if choice == '1':
        run_scanner_once()
    elif choice == '2':
        run_continuous_scanner(interval_minutes=15)
    elif choice == '3':
        run_continuous_scanner(interval_minutes=60)
    else:
        print("  Running single scan...")
        run_scanner_once()
    
    print("\n" + "#"*80)
    print("  📡 SCANNER COMPLETE")
    print("#"*80)