from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import ccxt
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PYDANTIC MODELS
# ============================================================

class AnalyzeRequest(BaseModel):
    coin: str
    capital: float = 1000
    position_status: str = "NO_POSITION"
    entry_price: Optional[float] = None
    risk_preference: str = "MEDIUM"
    reason_for_entry: Optional[str] = None  # BREAKOUT, DIP_BUY, FOMO, etc.

class SignalResponse(BaseModel):
    coin: str
    timestamp: str
    price: float
    price_source: str
    verdict: str
    confidence: str
    win_probability: float
    loss_probability: float
    signal_strength: str
    reasoning: List[str]
    warnings: List[str]
    risk: dict
    forecast: dict
    market_context: dict

class ScanResponse(BaseModel):
    timestamp: str
    market_context: dict
    signals: List[dict]
    buy_signals: List[dict]
    market_summary: str


# ============================================================
# PHASE 2: MARKET CONTEXT ENGINE
# ============================================================

class MarketContextEngine:
    """
    Analyzes broader market conditions:
    - BTC trend (for alt coin decisions)
    - Market regime (trending vs ranging)
    - Volatility state
    - Risk level
    """
    
    def __init__(self):
        self.exchange = ccxt.binance()
        self.btc_data = None
        self.last_update = None
        self.update_interval = 60  # Update every 60 seconds
    
    def fetch_btc_data(self):
        """Fetch BTC OHLCV for context analysis"""
        now = datetime.now().timestamp()
        
        if self.btc_data is not None and self.last_update:
            if now - self.last_update < self.update_interval:
                return self.btc_data
        
        try:
            # Fetch 1H data for short-term trend
            ohlcv_1h = self.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
            df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Fetch 4H data for medium-term trend
            ohlcv_4h = self.exchange.fetch_ohlcv('BTC/USDT', '4h', limit=50)
            df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Fetch 1D data for long-term trend
            ohlcv_1d = self.exchange.fetch_ohlcv('BTC/USDT', '1d', limit=30)
            df_1d = pd.DataFrame(ohlcv_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            self.btc_data = {
                '1h': df_1h,
                '4h': df_4h,
                '1d': df_1d
            }
            self.last_update = now
            
            return self.btc_data
            
        except Exception as e:
            print(f"Error fetching BTC data: {e}")
            return None
    
    def get_btc_context(self) -> dict:
        """
        Analyze BTC trend and conditions.
        
        Returns:
            {
                'trend_1h': 'UP' | 'DOWN' | 'SIDEWAYS',
                'trend_4h': 'UP' | 'DOWN' | 'SIDEWAYS',
                'trend_1d': 'UP' | 'DOWN' | 'SIDEWAYS',
                'overall_trend': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'strength': 0-100,
                'support_alts': True | False
            }
        """
        data = self.fetch_btc_data()
        
        if data is None:
            return {
                'trend_1h': 'UNKNOWN',
                'trend_4h': 'UNKNOWN',
                'trend_1d': 'UNKNOWN',
                'overall_trend': 'UNKNOWN',
                'strength': 50,
                'support_alts': True,
                'price': 0,
                'change_24h': 0
            }
        
        def get_trend(df):
            """Determine trend from OHLCV data"""
            if len(df) < 20:
                return 'SIDEWAYS', 50
            
            close = df['close'].values
            
            # Calculate EMAs
            ema_fast = pd.Series(close).ewm(span=9).mean().iloc[-1]
            ema_slow = pd.Series(close).ewm(span=21).mean().iloc[-1]
            
            # Calculate momentum
            returns = (close[-1] - close[-10]) / close[-10] * 100
            
            # Calculate strength based on momentum
            strength = min(100, abs(returns) * 10)
            
            if ema_fast > ema_slow and returns > 1:
                return 'UP', strength
            elif ema_fast < ema_slow and returns < -1:
                return 'DOWN', strength
            else:
                return 'SIDEWAYS', strength
        
        trend_1h, str_1h = get_trend(data['1h'])
        trend_4h, str_4h = get_trend(data['4h'])
        trend_1d, str_1d = get_trend(data['1d'])
        
        # Overall trend (weighted)
        trend_scores = {'UP': 1, 'SIDEWAYS': 0, 'DOWN': -1}
        weighted_score = (
            trend_scores.get(trend_1h, 0) * 0.2 +
            trend_scores.get(trend_4h, 0) * 0.3 +
            trend_scores.get(trend_1d, 0) * 0.5
        )
        
        if weighted_score > 0.3:
            overall_trend = 'BULLISH'
            support_alts = True
        elif weighted_score < -0.3:
            overall_trend = 'BEARISH'
            support_alts = False
        else:
            overall_trend = 'NEUTRAL'
            support_alts = True
        
        # Get current price and 24h change
        current_price = data['1h']['close'].iloc[-1]
        price_24h_ago = data['1h']['close'].iloc[-24] if len(data['1h']) >= 24 else data['1h']['close'].iloc[0]
        change_24h = (current_price - price_24h_ago) / price_24h_ago * 100
        
        return {
            'trend_1h': trend_1h,
            'trend_4h': trend_4h,
            'trend_1d': trend_1d,
            'overall_trend': overall_trend,
            'strength': round((str_1h + str_4h + str_1d) / 3, 1),
            'support_alts': support_alts,
            'price': round(current_price, 2),
            'change_24h': round(change_24h, 2)
        }
    
    def get_market_regime(self, coin_data: pd.DataFrame) -> dict:
        """
        Detect market regime: TRENDING vs RANGING.
        Uses simplified ADX-like calculation.
        """
        if coin_data is None or len(coin_data) < 30:
            return {
                'regime': 'UNKNOWN',
                'adx': None,
                'volatility': 'UNKNOWN',
                'volatility_pct': 0,
                'recommendation': '⚠️ Insufficient data - proceed with caution',
                'reliable': False
            }
        
        try:
            close = coin_data['close'].values[-30:]
            high = coin_data['high'].values[-30:]
            low = coin_data['low'].values[-30:]
            
            # Calculate ATR for volatility
            tr = []
            for i in range(1, len(close)):
                tr.append(max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                ))
            
            atr = np.mean(tr) if len(tr) > 0 else 0
            atr_pct = (atr / close[-1]) * 100 if close[-1] > 0 else 0
            
            # Calculate directional movement for ADX
            plus_dm = []
            minus_dm = []
            
            for i in range(1, len(high)):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                
                if up_move > down_move and up_move > 0:
                    plus_dm.append(up_move)
                else:
                    plus_dm.append(0)
                
                if down_move > up_move and down_move > 0:
                    minus_dm.append(down_move)
                else:
                    minus_dm.append(0)
            
            # Simplified ADX calculation
            avg_plus = np.mean(plus_dm[-14:]) if len(plus_dm) >= 14 else np.mean(plus_dm)
            avg_minus = np.mean(minus_dm[-14:]) if len(minus_dm) >= 14 else np.mean(minus_dm)
            
            if avg_plus + avg_minus > 0:
                dx = abs(avg_plus - avg_minus) / (avg_plus + avg_minus) * 100
                adx = dx
            else:
                adx = 0
            
            # ============================================
            # FIX: Handle unreliable ADX
            # ============================================
            if adx is None or np.isnan(adx) or adx < 5:
                return {
                    'regime': 'UNKNOWN',
                    'adx': round(adx, 1) if adx and not np.isnan(adx) else None,
                    'volatility': self._get_volatility_state(atr_pct),
                    'volatility_pct': round(atr_pct, 2),
                    'recommendation': '⚠️ Regime unclear - insufficient trend data',
                    'reliable': False
                }
            
            # Determine regime
            if adx > 25:
                regime = 'TRENDING'
                recommendation = '✅ Good for trend-following entries'
            elif adx < 15:
                regime = 'RANGING'
                recommendation = '⚠️ Avoid breakout trades, wait for range edges'
            else:
                regime = 'TRANSITIONING'
                recommendation = '⏳ Market direction unclear, reduce position size'
            
            return {
                'regime': regime,
                'adx': round(adx, 1),
                'volatility': self._get_volatility_state(atr_pct),
                'volatility_pct': round(atr_pct, 2),
                'recommendation': recommendation,
                'reliable': True
            }
            
        except Exception as e:
            print(f"Error in get_market_regime: {e}")
            return {
                'regime': 'UNKNOWN',
                'adx': None,
                'volatility': 'UNKNOWN',
                'volatility_pct': 0,
                'recommendation': '⚠️ Error calculating regime',
                'reliable': False
            }
    
    def _get_volatility_state(self, atr_pct: float) -> str:
        """Convert ATR percentage to volatility state"""
        if atr_pct > 5:
            return 'EXTREME'
        elif atr_pct > 3:
            return 'HIGH'
        elif atr_pct > 1.5:
            return 'NORMAL'
        elif atr_pct > 0:
            return 'LOW'
        else:
            return 'UNKNOWN'
        
        close = coin_data['close'].values[-30:]
        high = coin_data['high'].values[-30:]
        low = coin_data['low'].values[-30:]
        
        # Calculate ATR for volatility
        tr = np.maximum(high - low,
                       np.maximum(abs(high - np.roll(close, 1)),
                                 abs(low - np.roll(close, 1))))
        atr = np.mean(tr[1:])
        atr_pct = (atr / close[-1]) * 100
        
        # Calculate directional movement
        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)
        
        # Simplified ADX
        avg_plus = np.mean(plus_dm)
        avg_minus = np.mean(minus_dm)
        dx = abs(avg_plus - avg_minus) / (avg_plus + avg_minus + 0.0001) * 100
        adx = dx  # Simplified
        
        # Determine regime
        if adx > 25:
            regime = 'TRENDING'
            recommendation = 'Good for trend-following entries'
        elif adx < 15:
            regime = 'RANGING'
            recommendation = 'Avoid breakout trades, wait for range edges'
        else:
            regime = 'TRANSITIONING'
            recommendation = 'Market direction unclear, reduce position size'
        
        # Volatility state
        if atr_pct > 5:
            volatility = 'EXTREME'
        elif atr_pct > 3:
            volatility = 'HIGH'
        elif atr_pct > 1.5:
            volatility = 'NORMAL'
        else:
            volatility = 'LOW'
        
        return {
            'regime': regime,
            'adx': round(adx, 1),
            'volatility': volatility,
            'volatility_pct': round(atr_pct, 2),
            'recommendation': recommendation
        }
    
    def detect_market_shock(self, coin_data: pd.DataFrame) -> dict:
        """
        Detect market shock conditions (flash crash, extreme moves).
        """
        if coin_data is None or len(coin_data) < 10:
            return {'shock_detected': False, 'type': None, 'severity': 'NONE'}
        
        close = coin_data['close'].values
        
        # Check last 4 hours for extreme moves
        if len(close) >= 4:
            move_4h = (close[-1] - close[-4]) / close[-4] * 100
            
            if abs(move_4h) > 10:
                return {
                    'shock_detected': True,
                    'type': 'FLASH_CRASH' if move_4h < 0 else 'FLASH_PUMP',
                    'severity': 'EXTREME',
                    'move_pct': round(move_4h, 2),
                    'recommendation': '⚠️ PAUSE TRADING - Wait for stabilization'
                }
            elif abs(move_4h) > 5:
                return {
                    'shock_detected': True,
                    'type': 'HIGH_VOLATILITY',
                    'severity': 'HIGH',
                    'move_pct': round(move_4h, 2),
                    'recommendation': 'Reduce position sizes by 50%'
                }
        
        return {
            'shock_detected': False,
            'type': None,
            'severity': 'NONE',
            'move_pct': 0,
            'recommendation': 'Normal conditions'
        }


# ============================================================
# PHASE 2: ENHANCED WARNINGS ENGINE
# ============================================================

class WarningsEngine:
    """
    Generate smart warnings:
    - FOMO detection
    - Overextended price
    - BTC divergence
    - Crowded trades
    """
    
    def __init__(self):
        pass
    
    def check_all_warnings(self, coin: str, coin_data: pd.DataFrame, 
                           btc_context: dict, user_input: dict) -> List[dict]:
        """Check all warning conditions"""
        warnings = []
        
        # 1. FOMO Detection
        fomo_warning = self.check_fomo(user_input, coin_data)
        if fomo_warning:
            warnings.append(fomo_warning)
        
        # 2. Overextended Price
        overextended = self.check_overextended(coin_data)
        if overextended:
            warnings.append(overextended)
        
        # 3. BTC Divergence (for alts)
        if coin != 'BTC_USDT':
            btc_warning = self.check_btc_divergence(btc_context)
            if btc_warning:
                warnings.append(btc_warning)
        
        # 4. Time-based warnings
        time_warning = self.check_time_risk()
        if time_warning:
            warnings.append(time_warning)
        
        return warnings
    
    def check_fomo(self, user_input: dict, coin_data: pd.DataFrame) -> Optional[dict]:
        """Detect FOMO-driven entry"""
        reason = user_input.get('reason_for_entry', '')
        
        if reason == 'FOMO':
            return {
                'type': 'FOMO_DETECTED',
                'severity': 'HIGH',
                'message': '⚠️ FOMO entry detected - emotional trading is risky',
                'recommendation': 'Wait for pullback or use smaller position'
            }
        
        # Check if price already pumped significantly
        if coin_data is not None and len(coin_data) >= 24:
            close = coin_data['close'].values
            move_24h = (close[-1] - close[-24]) / close[-24] * 100
            
            if move_24h > 15:
                return {
                    'type': 'LATE_ENTRY',
                    'severity': 'MEDIUM',
                    'message': f'⚠️ Price already up {move_24h:.1f}% in 24h - late entry risk',
                    'recommendation': 'Wait for consolidation or pullback'
                }
        
        return None
    
    def check_overextended(self, coin_data: pd.DataFrame) -> Optional[dict]:
        """Check if price is overextended from moving averages"""
        if coin_data is None or len(coin_data) < 50:
            return None
        
        close = coin_data['close'].values
        
        # Calculate distance from 20 EMA
        ema_20 = pd.Series(close).ewm(span=20).mean().iloc[-1]
        distance = (close[-1] - ema_20) / ema_20 * 100
        
        if distance > 10:
            return {
                'type': 'OVEREXTENDED_UP',
                'severity': 'MEDIUM',
                'message': f'⚠️ Price {distance:.1f}% above EMA20 - overextended',
                'recommendation': 'High risk of pullback - wait or use tight stop'
            }
        elif distance < -10:
            return {
                'type': 'OVEREXTENDED_DOWN',
                'severity': 'LOW',
                'message': f'Price {abs(distance):.1f}% below EMA20 - potentially oversold',
                'recommendation': 'Could be good DIP entry if trend supports'
            }
        
        return None
    
    def check_btc_divergence(self, btc_context: dict) -> Optional[dict]:
        """Check if BTC trend is against alt trade"""
        if btc_context.get('overall_trend') == 'BEARISH':
            return {
                'type': 'BTC_BEARISH',
                'severity': 'HIGH',
                'message': '⚠️ BTC is bearish - alts typically underperform',
                'recommendation': 'Reduce alt exposure or wait for BTC recovery'
            }
        
        if not btc_context.get('support_alts', True):
            return {
                'type': 'BTC_NOT_SUPPORTIVE',
                'severity': 'MEDIUM',
                'message': 'BTC trend not supportive for alt trades',
                'recommendation': 'Consider BTC instead or reduce position'
            }
        
        return None
    
    def check_time_risk(self) -> Optional[dict]:
        """Check for risky trading times"""
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()
        
        # Weekend warning
        if weekday >= 5:
            return {
                'type': 'WEEKEND_TRADING',
                'severity': 'LOW',
                'message': 'Weekend trading - lower liquidity possible',
                'recommendation': 'Use wider stops, expect higher spreads'
            }
        
        # Low liquidity hours (rough estimate)
        if hour >= 0 and hour < 6:  # Midnight to 6 AM
            return {
                'type': 'LOW_LIQUIDITY_HOURS',
                'severity': 'LOW',
                'message': 'Low liquidity trading hours',
                'recommendation': 'Watch for wider spreads'
            }
        
        return None


# ============================================================
# LIVE PRICE FETCHER
# ============================================================

class LivePriceFetcher:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.price_cache = {}
        self.cache_time = {}
        self.cache_duration = 10
    
    def get_price(self, symbol: str) -> float:
        now = datetime.now().timestamp()
        
        if symbol in self.price_cache:
            if now - self.cache_time.get(symbol, 0) < self.cache_duration:
                return self.price_cache[symbol]
        
        try:
            ticker = self.exchange.fetch_ticker(symbol.replace('_', '/'))
            price = ticker['last']
            self.price_cache[symbol] = price
            self.cache_time[symbol] = now
            return price
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
    
    def get_all_prices(self) -> dict:
        coins = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']
        return {coin: self.get_price(coin) for coin in coins if self.get_price(coin)}


# ============================================================
# ENHANCED TRADING ENGINE V2
# ============================================================

class TradingEngineV2:
    """Enhanced Trading Engine with Phase 2 features"""
    
    def __init__(self):
        self.models = {}
        self.feature_cols = {}
        self.data_cache = {}
        
        self.price_fetcher = LivePriceFetcher()
        self.market_context = MarketContextEngine()
        self.warnings_engine = WarningsEngine()
        
        self.coins = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']
        
        self.thresholds = {
            'BTC_USDT': {'buy': 0.45, 'strong_buy': 0.50},
            'ETH_USDT': {'buy': 0.45, 'strong_buy': 0.50},
            'SOL_USDT': {'buy': 0.45, 'strong_buy': 0.50},
            'PEPE_USDT': {'buy': 0.50, 'strong_buy': 0.55}
        }
        
        self.volatility = {
            'BTC_USDT': {'sl': 0.03, 'tp': 0.05},
            'ETH_USDT': {'sl': 0.035, 'tp': 0.055},
            'SOL_USDT': {'sl': 0.04, 'tp': 0.06},
            'PEPE_USDT': {'sl': 0.05, 'tp': 0.08}
        }
        
        self.load_models()
        self.load_data()
    
    def load_models(self):
        for coin in self.coins:
            model_path = f"models/{coin}/decision_model.pkl"
            features_path = f"models/{coin}/decision_features.txt"
            
            if os.path.exists(model_path):
                self.models[coin] = joblib.load(model_path)
                with open(features_path, 'r') as f:
                    self.feature_cols[coin] = [line.strip() for line in f.readlines()]
                print(f"  ✅ {coin} model loaded")
    
    def load_data(self):
        for coin in self.coins:
            data_path = f"data/{coin}_multi_tf_features.csv"
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.data_cache[coin] = df
                print(f"  ✅ {coin} data loaded ({len(df):,} rows)")
    
    def get_probabilities(self, coin, features):
        if coin not in self.models:
            return None, None
        
        model = self.models[coin]
        feature_cols = self.feature_cols[coin]
        
        X = features[feature_cols].fillna(0)
        probas = model.predict_proba(X)[0]
        classes = list(model.classes_)
        
        win_idx = classes.index('WIN') if 'WIN' in classes else 0
        loss_idx = classes.index('LOSS') if 'LOSS' in classes else 1
        
        return probas[win_idx], probas[loss_idx]
    
    def analyze(self, coin: str, user_input: dict) -> dict:
        """Enhanced analysis with market context"""
        
        if coin not in self.coins:
            raise ValueError(f"Invalid coin: {coin}")
        
        # Get features
        features = self.data_cache[coin].tail(1)
        coin_data = self.data_cache[coin].tail(100)
        
        # Get probabilities
        win_prob, loss_prob = self.get_probabilities(coin, features)
        if win_prob is None:
            raise ValueError(f"Model not loaded for {coin}")
        
        # Get LIVE price
        live_price = self.price_fetcher.get_price(coin)
        current_price = live_price if live_price else features['close'].values[0]
        price_source = "LIVE" if live_price else "CACHED"
        
        # ============================================
        # PHASE 2: Get Market Context
        # ============================================
        btc_context = self.market_context.get_btc_context()
        regime = self.market_context.get_market_regime(coin_data)
        shock = self.market_context.detect_market_shock(coin_data)
        
        # ============================================
        # PHASE 2: Check Warnings
        # ============================================
        warnings_list = self.warnings_engine.check_all_warnings(
            coin, coin_data, btc_context, user_input
        )
        
        # Convert warnings to simple strings for response
        warning_messages = [w['message'] for w in warnings_list]
        
        # ============================================
        # DECISION LOGIC (Enhanced)
        # ============================================
        thresholds = self.thresholds[coin]
        position_status = user_input.get('position_status', 'NO_POSITION')
        
        reasoning = []
        
        # Check for blocking conditions
        if shock['shock_detected'] and shock['severity'] == 'EXTREME':
            return self._blocked_response(
                coin, current_price, price_source,
                "Market shock detected - trading paused",
                shock, btc_context, regime, warning_messages
            )
        
        # BTC context gate for alts
        if coin != 'BTC_USDT' and not btc_context.get('support_alts', True):
            # Reduce confidence for alt trades when BTC is bearish
            win_prob *= 0.7
            reasoning.append("WIN probability reduced due to bearish BTC")
        
        # Regime adjustment
        if regime['regime'] == 'RANGING' and win_prob < 0.55:
            reasoning.append(f"Market ranging (ADX: {regime['adx']}) - higher threshold needed")
            thresholds = {k: v + 0.05 for k, v in thresholds.items()}
        
        # Make decision
        if position_status == 'NO_POSITION':
            verdict, confidence, signal_strength = self._decide_entry(
                win_prob, loss_prob, thresholds, reasoning, warning_messages
            )
        else:
            entry_price = user_input.get('entry_price', current_price)
            verdict, confidence, signal_strength = self._decide_exit(
                win_prob, loss_prob, entry_price, current_price, reasoning, warning_messages
            )
        
        # Calculate risk
        risk = self._calculate_risk(
            verdict, confidence, win_prob,
            user_input.get('capital', 1000),
            user_input.get('risk_preference', 'MEDIUM'),
            coin, current_price, regime, shock
        )
        
        # Forecast
        forecast = self._get_forecast(win_prob, loss_prob, coin, current_price)
        
        return {
            'coin': coin,
            'timestamp': datetime.now().isoformat(),
            'price': float(current_price),
            'price_source': price_source,
            'verdict': verdict,
            'confidence': confidence,
            'win_probability': round(float(win_prob), 4),
            'loss_probability': round(float(loss_prob), 4),
            'signal_strength': signal_strength,
            'reasoning': reasoning,
            'warnings': warning_messages,
            'risk': risk,
            'forecast': forecast,
            'market_context': {
                'btc': btc_context,
                'regime': regime,
                'shock': shock
            }
        }
    
    def _blocked_response(self, coin, price, price_source, reason, shock, btc, regime, warnings):
        """Return blocked response for extreme conditions"""
        return {
            'coin': coin,
            'timestamp': datetime.now().isoformat(),
            'price': float(price),
            'price_source': price_source,
            'verdict': 'BLOCKED',
            'confidence': 'HIGH',
            'win_probability': 0,
            'loss_probability': 0,
            'signal_strength': 'DANGER',
            'reasoning': [reason, shock.get('recommendation', '')],
            'warnings': warnings + [f"🚨 {shock.get('type', 'UNKNOWN')}: {shock.get('recommendation', '')}"],
            'risk': {'action': 'NO_TRADE', 'reason': 'Market conditions too risky'},
            'forecast': {'direction': 'UNKNOWN'},
            'market_context': {'btc': btc, 'regime': regime, 'shock': shock}
        }
    
    def _decide_entry(self, win_prob, loss_prob, thresholds, reasoning, warnings):
        if win_prob >= thresholds['strong_buy']:
            reasoning.append(f"High WIN probability: {win_prob*100:.1f}%")
            reasoning.append("Strong entry signal detected")
            return 'BUY', 'HIGH', 'STRONG'
        
        elif win_prob >= thresholds['buy']:
            reasoning.append(f"Moderate WIN probability: {win_prob*100:.1f}%")
            if loss_prob < 0.35:
                reasoning.append("LOSS probability acceptable")
                return 'BUY', 'MEDIUM', 'MODERATE'
            else:
                warnings.append("Elevated LOSS probability")
                return 'WAIT', 'MEDIUM', 'WEAK'
        
        elif loss_prob >= 0.50:
            reasoning.append(f"High LOSS probability: {loss_prob*100:.1f}%")
            warnings.append("⚠️ High risk of loss - avoid entry")
            return 'AVOID', 'HIGH', 'DANGER'
        
        else:
            reasoning.append(f"WIN: {win_prob*100:.1f}%, LOSS: {loss_prob*100:.1f}%")
            reasoning.append("No clear edge")
            return 'WAIT', 'LOW', 'NEUTRAL'
    
    def _decide_exit(self, win_prob, loss_prob, entry_price, current_price, reasoning, warnings):
        pnl_pct = (current_price - entry_price) / entry_price * 100
        reasoning.append(f"Current P&L: {pnl_pct:+.2f}%")
        
        if loss_prob >= 0.50:
            reasoning.append(f"High LOSS probability: {loss_prob*100:.1f}%")
            return 'EXIT', 'HIGH', 'DANGER'
        elif win_prob >= 0.40:
            reasoning.append(f"WIN probability favorable: {win_prob*100:.1f}%")
            return 'HOLD', 'MEDIUM', 'MODERATE'
        elif pnl_pct > 3:
            reasoning.append("In profit, weakening signal")
            return 'HOLD', 'LOW', 'WEAK'
        elif pnl_pct < -2:
            reasoning.append("In loss, weak signal")
            return 'EXIT', 'MEDIUM', 'WEAK'
        else:
            reasoning.append("No clear signal")
            return 'HOLD', 'LOW', 'NEUTRAL'
    
    def _calculate_risk(self, verdict, confidence, win_prob, capital, risk_pref, coin, price, regime, shock):
        if verdict in ['WAIT', 'EXIT', 'AVOID', 'BLOCKED']:
            return {
                'position_size_pct': 0,
                'position_size_usd': 0,
                'stop_loss_pct': 0,
                'stop_loss_price': 0,
                'take_profit_pct': 0,
                'take_profit_price': 0,
                'max_loss_usd': 0,
                'action': 'NO_TRADE'
            }
        
        base_size = {'HIGH': 0.40, 'MEDIUM': 0.25, 'LOW': 0.15}.get(confidence, 0.20)
        risk_mult = {'LOW': 0.6, 'MEDIUM': 1.0, 'HIGH': 1.4}.get(risk_pref, 1.0)
        
        # Reduce size in high volatility or shock conditions
        if shock.get('severity') == 'HIGH':
            risk_mult *= 0.5
        if regime.get('volatility') == 'EXTREME':
            risk_mult *= 0.5
        elif regime.get('volatility') == 'HIGH':
            risk_mult *= 0.75
        
        position_size_pct = min(base_size * risk_mult, 0.50)
        position_size_usd = capital * position_size_pct
        
        vol = self.volatility.get(coin, {'sl': 0.03, 'tp': 0.05})
        
        # Adjust SL/TP for volatility
        vol_mult = {'EXTREME': 1.5, 'HIGH': 1.25, 'NORMAL': 1.0, 'LOW': 0.8}.get(
            regime.get('volatility', 'NORMAL'), 1.0
        )
        
        stop_loss_pct = vol['sl'] * vol_mult
        take_profit_pct = vol['tp'] * vol_mult
        
        stop_loss_price = price * (1 - stop_loss_pct)
        take_profit_price = price * (1 + take_profit_pct)
        max_loss_usd = position_size_usd * stop_loss_pct
        
        return {
            'position_size_pct': round(position_size_pct * 100, 1),
            'position_size_usd': round(position_size_usd, 2),
            'stop_loss_pct': round(stop_loss_pct * 100, 2),
            'stop_loss_price': round(stop_loss_price, 8 if price < 0.01 else 2),
            'take_profit_pct': round(take_profit_pct * 100, 2),
            'take_profit_price': round(take_profit_price, 8 if price < 0.01 else 2),
            'max_loss_usd': round(max_loss_usd, 2),
            'risk_level': regime.get('volatility', 'NORMAL'),
            'action': 'OPEN_POSITION'
        }
    
    def _get_forecast(self, win_prob, loss_prob, coin, price):
        sideways_prob = max(0, 1 - win_prob - loss_prob)
        
        moves = {
            'BTC_USDT': {'up': 5, 'down': 4},
            'ETH_USDT': {'up': 7, 'down': 5},
            'SOL_USDT': {'up': 10, 'down': 8},
            'PEPE_USDT': {'up': 15, 'down': 12}
        }.get(coin, {'up': 5, 'down': 4})
        
        if win_prob > loss_prob and win_prob > sideways_prob:
            direction = 'BULLISH'
        elif loss_prob > win_prob and loss_prob > sideways_prob:
            direction = 'BEARISH'
        else:
            direction = 'SIDEWAYS'
        
        return {
            'direction': direction,
            'current_price': price,
            'bull_target': round(price * (1 + moves['up']/100), 8 if price < 0.01 else 2),
            'bear_target': round(price * (1 - moves['down']/100), 8 if price < 0.01 else 2),
            'probabilities': {
                'up': round(win_prob, 3),
                'sideways': round(sideways_prob, 3),
                'down': round(loss_prob, 3)
            }
        }
    
    def scan_all(self) -> dict:
        """Scan all coins with full context"""
        signals = []
        buy_signals = []
        
        # Get BTC context once
        btc_context = self.market_context.get_btc_context()
        
        for coin in self.coins:
            try:
                result = self.analyze(coin, {'position_status': 'NO_POSITION'})
                
                signal = {
                    'coin': coin,
                    'price': result['price'],
                    'price_source': result['price_source'],
                    'win_probability': result['win_probability'],
                    'loss_probability': result['loss_probability'],
                    'verdict': result['verdict'],
                    'confidence': result['confidence'],
                    'signal_strength': result['signal_strength'],
                    'warnings_count': len(result['warnings'])
                }
                
                signals.append(signal)
                
                if result['verdict'] == 'BUY':
                    buy_signals.append(signal)
                    
            except Exception as e:
                print(f"Error scanning {coin}: {e}")
        
        signals = sorted(signals, key=lambda x: x['win_probability'], reverse=True)
        
        # Enhanced market summary
        if any(s['verdict'] == 'BLOCKED' for s in signals):
            market_summary = "🚨 MARKET SHOCK - Trading paused for safety"
        elif buy_signals:
            market_summary = f"🟢 {len(buy_signals)} BUY signal(s) detected!"
        elif any(s['verdict'] == 'AVOID' for s in signals):
            market_summary = "🔴 High risk conditions - avoid trading"
        elif btc_context.get('overall_trend') == 'BEARISH':
            market_summary = "⚠️ BTC bearish - alt trades risky"
        else:
            market_summary = "⏳ No buy signals - waiting for better conditions"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'market_context': {
                'btc': btc_context,
                'overall_risk': 'HIGH' if any(s['verdict'] in ['BLOCKED', 'AVOID'] for s in signals) else 'NORMAL'
            },
            'signals': signals,
            'buy_signals': buy_signals,
            'market_summary': market_summary
        }


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Crypto AI Trading API v2",
    description="AI-powered trading signals with BTC context, regime detection, and smart warnings",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "="*60)
print("  🚀 Initializing Trading Engine V2 (Phase 2)...")
print("="*60)
engine = TradingEngineV2()
print("="*60 + "\n")


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "Crypto AI Trading API",
        "version": "2.2.0 (Phase 2)",
        "features": [
            "Live prices",
            "BTC context analysis",
            "Market regime detection",
            "Smart warnings (FOMO, overextended, etc.)",
            "Market shock detection"
        ],
        "endpoints": ["/scan", "/analyze/{coin}", "/context", "/health", "/docs"]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": {coin: coin in engine.models for coin in engine.coins},
        "features": ["live_prices", "btc_context", "regime_detection", "warnings"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/scan")
async def scan():
    """Scan all coins with full market context"""
    return engine.scan_all()

@app.get("/analyze/{coin}")
async def analyze(
    coin: str, 
    capital: float = 1000, 
    risk: str = "MEDIUM",
    reason: str = None
):
    """Analyze a specific coin with context"""
    try:
        return engine.analyze(coin, {
            'capital': capital,
            'position_status': 'NO_POSITION',
            'risk_preference': risk,
            'reason_for_entry': reason
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/context")
async def get_context():
    """Get current market context (BTC trend, regime, etc.)"""
    return {
        'timestamp': datetime.now().isoformat(),
        'btc': engine.market_context.get_btc_context(),
        'overall': 'Data available for all coins'
    }

@app.get("/context/{coin}")
async def get_coin_context(coin: str):
    """Get market context for a specific coin"""
    if coin not in engine.data_cache:
        raise HTTPException(status_code=404, detail=f"Coin {coin} not found")
    
    coin_data = engine.data_cache[coin].tail(100)
    
    return {
        'coin': coin,
        'timestamp': datetime.now().isoformat(),
        'btc': engine.market_context.get_btc_context(),
        'regime': engine.market_context.get_market_regime(coin_data),
        'shock': engine.market_context.detect_market_shock(coin_data)
    }

@app.get("/prices")
async def get_prices():
    """Get all live prices"""
    return engine.price_fetcher.get_all_prices()


if __name__ == "__main__":
    import uvicorn
    print("\n" + "#"*60)
    print("  🚀 CRYPTO AI TRADING API v2.2 (Phase 2)")
    print("#"*60)
    print("\n  Features:")
    print("    ✅ Live prices from Binance")
    print("    ✅ BTC context analysis")
    print("    ✅ Market regime detection")
    print("    ✅ Smart warnings (FOMO, overextended)")
    print("    ✅ Market shock detection")
    print("\n  Server: http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("\n" + "#"*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)