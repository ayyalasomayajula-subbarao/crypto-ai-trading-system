"""
Crypto AI Trading API - v4.1
With Scenario Engine + WebSocket + Fixed Metrics

FIXED in v4.1:
- Replaced confusing "Edge" with clear "Expectancy" and "Readiness"
- Expectancy = WIN% - LOSS% (is trade profitable?)
- Readiness = WIN% - Threshold% (how close to entry?)
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Set
import pandas as pd
import numpy as np
import joblib
import ccxt
import os
from datetime import datetime
from enum import Enum
import asyncio
import websockets
import json
import warnings
warnings.filterwarnings('ignore')

# Import Scenario Engine (SEPARATE MODULE)
from engines.scenario_engine import ScenarioEngine, ScenarioDecision


# ============================================================
# ENUMS
# ============================================================

class TradeType(str, Enum):
    SCALP = "SCALP"
    SHORT_TERM = "SHORT_TERM"
    SWING = "SWING"
    INVESTMENT = "INVESTMENT"

class HoldingStatus(str, Enum):
    NO_POSITION = "NO_POSITION"
    IN_POSITION = "IN_POSITION"

class RiskPreference(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class ExperienceLevel(str, Enum):
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"

class TradeReason(str, Enum):
    STRATEGY = "STRATEGY"
    FOMO = "FOMO"
    NEWS = "NEWS"
    TIP = "TIP"
    DIP_BUY = "DIP_BUY"


# ============================================================
# REQUEST MODEL
# ============================================================

class AnalyzeRequest(BaseModel):
    coin: str
    capital: float = 1000
    trade_type: TradeType = TradeType.SWING
    entry_price: Optional[float] = None
    holding_status: HoldingStatus = HoldingStatus.NO_POSITION
    risk_preference: RiskPreference = RiskPreference.MEDIUM
    experience_level: ExperienceLevel = ExperienceLevel.INTERMEDIATE
    reason_for_trade: Optional[TradeReason] = None
    recent_losses: int = 0
    trades_today: int = 0
    max_daily_trades: int = 5


# ============================================================
# CONFIGURATION
# ============================================================

TRADE_TYPE_CONFIG = {
    TradeType.SCALP: {
        "min_win_prob": 0.55, "max_loss_prob": 0.40, "adx_required": 25,
        "tp_pct": 0.02, "sl_pct": 0.01, "position_mult": 0.8, "max_hold_hours": 4
    },
    TradeType.SHORT_TERM: {
        "min_win_prob": 0.50, "max_loss_prob": 0.42, "adx_required": 22,
        "tp_pct": 0.04, "sl_pct": 0.02, "position_mult": 0.9, "max_hold_hours": 48
    },
    TradeType.SWING: {
        "min_win_prob": 0.45, "max_loss_prob": 0.45, "adx_required": 18,
        "tp_pct": 0.08, "sl_pct": 0.04, "position_mult": 1.0, "max_hold_hours": 168
    },
    TradeType.INVESTMENT: {
        "min_win_prob": 0.35, "max_loss_prob": 0.50, "adx_required": 12,
        "tp_pct": 0.25, "sl_pct": 0.12, "position_mult": 0.6, "max_hold_hours": 720
    }
}

COIN_VOLATILITY = {
    "BTC_USDT": 1.0,
    "ETH_USDT": 1.2,
    "SOL_USDT": 1.5,
    "PEPE_USDT": 2.0
}

EXPERIENCE_CONFIG = {
    ExperienceLevel.BEGINNER: {
        "threshold_boost": 0.10,
        "position_mult": 0.5,
        "block_scalps": True
    },
    ExperienceLevel.INTERMEDIATE: {
        "threshold_boost": 0.05,
        "position_mult": 0.8,
        "block_scalps": False
    },
    ExperienceLevel.ADVANCED: {
        "threshold_boost": 0.0,
        "position_mult": 1.0,
        "block_scalps": False
    }
}


# ============================================================
# LIVE PRICE FETCHER
# ============================================================

class LivePriceFetcher:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.cache = {}
        self.cache_time = {}
    
    def get_price(self, symbol: str) -> Optional[float]:
        now = datetime.now().timestamp()
        if symbol in self.cache and now - self.cache_time.get(symbol, 0) < 10:
            return self.cache[symbol]
        try:
            ticker = self.exchange.fetch_ticker(symbol.replace('_', '/'))
            self.cache[symbol] = ticker['last']
            self.cache_time[symbol] = now
            return ticker['last']
        except:
            return None


# ============================================================
# MARKET CONTEXT ENGINE
# ============================================================

class MarketContextEngine:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.btc_cache = None
        self.btc_cache_time = None
    
    def get_btc_context(self) -> dict:
        try:
            now = datetime.now().timestamp()
            if self.btc_cache and self.btc_cache_time and now - self.btc_cache_time < 60:
                return self.btc_cache
            
            df_1h = pd.DataFrame(
                self.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=50),
                columns=['ts', 'o', 'h', 'l', 'c', 'v']
            )
            df_4h = pd.DataFrame(
                self.exchange.fetch_ohlcv('BTC/USDT', '4h', limit=30),
                columns=['ts', 'o', 'h', 'l', 'c', 'v']
            )
            df_1d = pd.DataFrame(
                self.exchange.fetch_ohlcv('BTC/USDT', '1d', limit=14),
                columns=['ts', 'o', 'h', 'l', 'c', 'v']
            )
            
            def get_trend(close):
                ema9 = pd.Series(close).ewm(span=9).mean().iloc[-1]
                ema21 = pd.Series(close).ewm(span=21).mean().iloc[-1]
                ret = (close[-1] - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
                if ema9 > ema21 and ret > 0.5:
                    return 'UP'
                elif ema9 < ema21 and ret < -0.5:
                    return 'DOWN'
                return 'SIDEWAYS'
            
            trend_1h = get_trend(df_1h['c'].values)
            trend_4h = get_trend(df_4h['c'].values)
            trend_1d = get_trend(df_1d['c'].values)
            
            scores = {'UP': 1, 'SIDEWAYS': 0, 'DOWN': -1}
            weighted = scores[trend_1h]*0.2 + scores[trend_4h]*0.3 + scores[trend_1d]*0.5
            
            if weighted > 0.3:
                overall = 'BULLISH'
                support_alts = True
            elif weighted < -0.3:
                overall = 'BEARISH'
                support_alts = False
            else:
                overall = 'NEUTRAL'
                support_alts = True
            
            price = df_1h['c'].iloc[-1]
            change_24h = (price - df_1h['c'].iloc[-24]) / df_1h['c'].iloc[-24] * 100 if len(df_1h) >= 24 else 0
            change_1h = (price - df_1h['c'].iloc[-2]) / df_1h['c'].iloc[-2] * 100 if len(df_1h) >= 2 else 0
            
            # Calculate trend strength
            def get_strength(df):
                if len(df) < 10:
                    return 50
                close = df['c'].values
                returns = (close[-1] - close[-10]) / close[-10] * 100
                return min(100, abs(returns) * 10)

            str_1h = get_strength(df_1h)
            str_4h = get_strength(df_4h)
            str_1d = get_strength(df_1d)

            self.btc_cache = {
                'trend_1h': trend_1h,
                'trend_4h': trend_4h,
                'trend_1d': trend_1d,
                'overall_trend': overall,
                'strength': round((str_1h + str_4h + str_1d) / 3, 1),
                'support_alts': support_alts,
                'price': round(price, 2),
                'change_24h': round(change_24h, 2),
                'change_1h': round(change_1h, 2)
            }
            self.btc_cache_time = now
            return self.btc_cache
            
        except Exception as e:
            print(f"BTC context error: {e}")
            return {
                'trend_1h': 'UNKNOWN', 'trend_4h': 'UNKNOWN', 'trend_1d': 'UNKNOWN',
                'overall_trend': 'UNKNOWN', 'strength': 50, 'support_alts': True,
                'price': 0, 'change_24h': 0, 'change_1h': 0
            }
    
    def get_regime(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) < 30:
            return {'regime': 'UNKNOWN', 'adx': 0, 'volatility': 'NORMAL', 'reliable': False}
        
        try:
            close = df['close'].values[-30:]
            high = df['high'].values[-30:]
            low = df['low'].values[-30:]
            
            tr = [max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1])) for i in range(1, len(close))]
            atr = np.mean(tr)
            atr_pct = (atr / close[-1]) * 100
            
            plus_dm = [max(high[i]-high[i-1], 0) if high[i]-high[i-1] > low[i-1]-low[i] else 0 for i in range(1, len(high))]
            minus_dm = [max(low[i-1]-low[i], 0) if low[i-1]-low[i] > high[i]-high[i-1] else 0 for i in range(1, len(low))]
            
            avg_plus = np.mean(plus_dm[-14:])
            avg_minus = np.mean(minus_dm[-14:])
            adx = abs(avg_plus - avg_minus) / (avg_plus + avg_minus + 0.0001) * 100
            
            if adx < 5 or np.isnan(adx):
                return {'regime': 'UNKNOWN', 'adx': 0, 'volatility': 'UNKNOWN', 'reliable': False,
                        'recommendation': '⚠️ Insufficient trend data'}
            
            regime = 'TRENDING' if adx > 25 else ('RANGING' if adx < 15 else 'TRANSITIONING')
            volatility = 'EXTREME' if atr_pct > 5 else ('HIGH' if atr_pct > 3 else ('NORMAL' if atr_pct > 1.5 else 'LOW'))
            
            recommendations = {
                'TRENDING': '✅ Good for trend-following entries',
                'RANGING': '⚠️ Avoid breakouts, wait for range edges',
                'TRANSITIONING': '⏳ Market direction unclear, reduce exposure'
            }
            
            return {
                'regime': regime,
                'adx': round(adx, 1),
                'volatility': volatility,
                'volatility_pct': round(atr_pct, 2),
                'recommendation': recommendations.get(regime, ''),
                'reliable': True
            }
        except:
            return {'regime': 'UNKNOWN', 'adx': 0, 'volatility': 'NORMAL', 'reliable': False}


# ============================================================
# WEBSOCKET PRICE STREAMER
# ============================================================

class PriceStreamer:
    """
    Connects to Binance WebSocket and broadcasts prices to all clients.
    """
    
    def __init__(self):
        self.clients: Set[WebSocket] = set()
        self.prices: dict = {}
        self.previous_prices: dict = {}
        self.running = False
        
        self.coins = {
            'BTC_USDT': 'btcusdt',
            'ETH_USDT': 'ethusdt',
            'SOL_USDT': 'solusdt',
            'PEPE_USDT': 'pepeusdt'
        }
    
    async def connect_client(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.add(websocket)
        if self.prices:
            await websocket.send_json({
                'type': 'initial',
                'prices': self.prices
            })
        print(f"  📱 Client connected. Total: {len(self.clients)}")
    
    def disconnect_client(self, websocket: WebSocket):
        self.clients.discard(websocket)
        print(f"  📱 Client disconnected. Total: {len(self.clients)}")
    
    async def broadcast(self, data: dict):
        dead_clients = set()
        for client in self.clients:
            try:
                await client.send_json(data)
            except:
                dead_clients.add(client)
        self.clients -= dead_clients
    
    async def binance_listener(self, symbol: str, coin_key: str):
        url = f"wss://stream.binance.com:9443/ws/{symbol}@ticker"
        
        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    print(f"  🔗 Connected to Binance: {coin_key}")
                    
                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)
                            
                            price = float(data['c'])
                            change_24h = float(data['P'])
                            
                            self.previous_prices[coin_key] = self.prices.get(coin_key, {}).get('price', price)
                            
                            self.prices[coin_key] = {
                                'price': price,
                                'change_24h': round(change_24h, 2),
                                'timestamp': datetime.now().isoformat(),
                                'direction': 'up' if price > self.previous_prices[coin_key] else ('down' if price < self.previous_prices[coin_key] else 'same')
                            }
                            
                            await self.broadcast({
                                'type': 'price_update',
                                'coin': coin_key,
                                'data': self.prices[coin_key]
                            })
                            
                        except asyncio.TimeoutError:
                            await ws.ping()
                            
            except Exception as e:
                print(f"  ❌ Binance connection error ({coin_key}): {e}")
                await asyncio.sleep(5)
    
    async def start(self):
        self.running = True
        tasks = [
            self.binance_listener(symbol, coin_key)
            for coin_key, symbol in self.coins.items()
        ]
        await asyncio.gather(*tasks)
    
    def stop(self):
        self.running = False


# Create global price streamer
price_streamer = PriceStreamer()


# ============================================================
# TRADING ENGINE (Orchestrates All Components)
# ============================================================

class TradingEngine:
    def __init__(self):
        self.models = {}
        self.feature_cols = {}
        self.data_cache = {}
        
        self.price_fetcher = LivePriceFetcher()
        self.market_context = MarketContextEngine()
        self.scenario_engine = ScenarioEngine()
        
        self.coins = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']
        self.load_all()
    
    def load_all(self):
        for coin in self.coins:
            model_path = f"models/{coin}/decision_model.pkl"
            features_path = f"models/{coin}/decision_features.txt"
            if os.path.exists(model_path):
                self.models[coin] = joblib.load(model_path)
                with open(features_path, 'r') as f:
                    self.feature_cols[coin] = [l.strip() for l in f.readlines()]
                print(f"  ✅ {coin} model loaded")
            
            data_path = f"data/{coin}_multi_tf_features.csv"
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.data_cache[coin] = df
    
    def get_probabilities(self, coin, features):
        if coin not in self.models:
            return None, None
        model = self.models[coin]
        X = features[self.feature_cols[coin]].fillna(0)
        probas = model.predict_proba(X)[0]
        classes = list(model.classes_)
        win_idx = classes.index('WIN')
        loss_idx = classes.index('LOSS')
        return probas[win_idx], probas[loss_idx]
    
    def analyze(self, request: AnalyzeRequest) -> dict:
        """
        Main analysis function with FIXED metrics:
        - Expectancy: WIN% - LOSS% (is trade profitable?)
        - Readiness: WIN% - Threshold% (how close to entry?)
        """
        
        coin = request.coin
        if coin not in self.coins:
            raise ValueError(f"Invalid coin: {coin}")
        
        # Get data
        features = self.data_cache[coin].tail(1)
        coin_data = self.data_cache[coin].tail(100)
        
        # Get prices
        ws_price = price_streamer.prices.get(coin, {}).get('price')
        live_price = ws_price if ws_price else self.price_fetcher.get_price(coin)
        current_price = live_price if live_price else features['close'].values[0]
        entry_price = request.entry_price if request.entry_price else current_price
        
        # Get market context
        btc_context = self.market_context.get_btc_context()
        regime = self.market_context.get_regime(coin_data)
        
        # ============================================
        # STEP 1: SCENARIO ENGINE (Gate Layer)
        # ============================================
        scenario_result = self.scenario_engine.evaluate(
            btc_data=btc_context,
            coin=coin,
            coin_data=coin_data,
            user_context={
                'reason_for_trade': request.reason_for_trade.value if request.reason_for_trade else None,
                'recent_losses': request.recent_losses,
                'trades_today': request.trades_today,
                'max_daily_trades': request.max_daily_trades
            }
        )
        
        # If BLOCKED → Stop here
        if scenario_result.decision == ScenarioDecision.BLOCK:
            return {
                'coin': coin,
                'timestamp': datetime.now().isoformat(),
                'price': current_price,
                'price_source': 'WEBSOCKET' if ws_price else ('LIVE' if live_price else 'CACHED'),
                
                'capital': request.capital,
                'trade_type': request.trade_type.value,
                'experience_level': request.experience_level.value,
                
                'verdict': 'BLOCKED',
                'confidence': 'HIGH',
                'blocked_by': 'SCENARIO_ENGINE',
                'block_reason': scenario_result.block_reason,
                
                'active_scenarios': [s.dict() for s in scenario_result.active_scenarios],
                'scenario_count': len(scenario_result.active_scenarios),
                
                'model_ran': False,
                'win_probability': None,
                'loss_probability': None,
                
                # Metrics are None when blocked
                'expectancy': None,
                'expectancy_status': None,
                'readiness': None,
                'readiness_status': None,
                
                'risk': {'action': 'NO_TRADE', 'position_size_usd': 0},
                
                'suggested_action': {
                    'action': 'STOP',
                    'message': scenario_result.block_reason,
                    'next_check': scenario_result.resume_check,
                    'conditions': ['Resolve active scenario(s) first']
                },
                
                'market_context': {
                    'btc': btc_context,
                    'regime': regime
                }
            }
        
        # ============================================
        # STEP 2: DECISION ENGINE (Model Logic)
        # ============================================
        
        win_prob, loss_prob = self.get_probabilities(coin, features)
        if win_prob is None:
            raise ValueError(f"Model not loaded for {coin}")
        
        # Get configs
        trade_config = TRADE_TYPE_CONFIG[request.trade_type]
        exp_config = EXPERIENCE_CONFIG[request.experience_level]
        
        # Apply scenario modifications
        scenario_position_mult = scenario_result.position_size_multiplier
        scenario_threshold_boost = scenario_result.threshold_boost
        
        # Calculate thresholds
        base_threshold = trade_config['min_win_prob']
        adjusted_threshold = base_threshold + exp_config['threshold_boost'] + scenario_threshold_boost
        
        # ============================================
        # NEW: Calculate EXPECTANCY and READINESS
        # ============================================
        
        # EXPECTANCY: WIN% - LOSS% (is this trade profitable?)
        # Answers: "Am I more likely to win or lose?"
        expectancy = (win_prob - loss_prob) * 100
        
        if expectancy > 5:
            expectancy_status = '✅ Positive (WIN > LOSS)'
        elif expectancy > 0:
            expectancy_status = '⚠️ Slightly positive'
        elif expectancy > -5:
            expectancy_status = '⚠️ Slightly negative'
        else:
            expectancy_status = '❌ Negative (LOSS > WIN)'
        
        # READINESS: WIN% - Threshold% (how close to entry?)
        # Answers: "How close am I to meeting entry requirements?"
        readiness = (win_prob - adjusted_threshold) * 100
        
        if readiness > 10:
            readiness_status = '✅ Well above threshold'
        elif readiness > 0:
            readiness_status = '✅ Above threshold (ready)'
        elif readiness > -5:
            readiness_status = '⚠️ Near threshold (almost ready)'
        else:
            readiness_status = '❌ Below threshold (not ready)'
        
        # Risk-adjusted EV (for advanced users / position sizing)
        vol_mult = COIN_VOLATILITY.get(coin, 1.0)
        tp_pct = trade_config['tp_pct'] * vol_mult
        sl_pct = trade_config['sl_pct'] * vol_mult
        risk_adjusted_ev = (win_prob * tp_pct * 100) - (loss_prob * sl_pct * 100)
        
        # ============================================
        # DECISION LOGIC (using clear metrics)
        # ============================================
        reasoning = []
        warnings = []
        
        # Add scenario warnings
        for scenario in scenario_result.active_scenarios:
            warnings.append(f"{scenario.icon} {scenario.message}")
        
        # Check if beginner trying scalp
        if exp_config['block_scalps'] and request.trade_type == TradeType.SCALP:
            verdict = 'WAIT'
            confidence = 'HIGH'
            reasoning.append("Scalp trading not recommended for beginners")
            warnings.append("🛡️ Start with SWING trades to build experience")
        
        # Check regime
        elif regime.get('reliable', False) and regime.get('adx', 0) < trade_config['adx_required']:
            verdict = 'WAIT'
            confidence = 'MEDIUM'
            reasoning.append(f"ADX ({regime['adx']}) below required ({trade_config['adx_required']}) for {request.trade_type.value}")
        
        # Check BTC context for alts
        elif coin != 'BTC_USDT' and not btc_context.get('support_alts', True):
            verdict = 'WAIT'
            confidence = 'MEDIUM'
            reasoning.append("BTC is bearish – alts typically underperform")
        
        # Check EXPECTANCY first (most important)
        elif expectancy < -10:
            verdict = 'AVOID'
            confidence = 'HIGH'
            reasoning.append(f"Expectancy strongly negative: {expectancy:.1f}%")
            reasoning.append("LOSS significantly more likely than WIN")
        
        # Check LOSS probability
        elif loss_prob > 0.50:
            verdict = 'AVOID'
            confidence = 'HIGH'
            reasoning.append(f"High LOSS probability: {loss_prob*100:.1f}%")
        
        # Check both READINESS and EXPECTANCY
        elif readiness >= 0 and expectancy >= 0:
            # Both positive = BUY
            verdict = 'BUY'
            confidence = 'HIGH' if readiness >= 5 and expectancy >= 5 else 'MEDIUM'
            reasoning.append(f"Expectancy positive: {expectancy:.1f}%")
            reasoning.append(f"WIN ({win_prob*100:.1f}%) above threshold ({adjusted_threshold*100:.0f}%)")
        
        elif readiness >= 0 and expectancy < 0:
            # Meets threshold but expectancy negative - THE CONFUSING CASE WE FIXED!
            verdict = 'WAIT'
            confidence = 'MEDIUM'
            reasoning.append(f"Above threshold but expectancy negative: {expectancy:.1f}%")
            reasoning.append("Threshold met, but LOSS > WIN - wait for better setup")
        
        elif readiness >= -5:
            # Near threshold
            verdict = 'WAIT'
            confidence = 'MEDIUM' if expectancy >= 0 else 'LOW'
            reasoning.append(f"Near threshold (readiness: {readiness:.1f}%)")
            if expectancy < 0:
                reasoning.append(f"Expectancy negative: {expectancy:.1f}%")
        
        else:
            verdict = 'WAIT'
            confidence = 'LOW'
            reasoning.append(f"Below threshold (readiness: {readiness:.1f}%)")
            reasoning.append(f"Expectancy: {expectancy:.1f}%")
        
        # ============================================
        # STEP 3: RISK ENGINE
        # ============================================
        
        if verdict in ['AVOID', 'WAIT', 'BLOCKED']:
            risk = {'action': 'NO_TRADE', 'position_size_usd': 0}
        else:
            pos_mult = (trade_config['position_mult'] * 
                       exp_config['position_mult'] * 
                       scenario_position_mult)
            
            pos_pct = min(pos_mult * 0.3, 0.5)
            pos_usd = request.capital * pos_pct
            
            risk = {
                'action': 'OPEN_POSITION',
                'position_size_pct': round(pos_pct * 100, 1),
                'position_size_usd': round(pos_usd, 2),
                'entry_price': round(entry_price, 8 if entry_price < 0.01 else 2),
                'stop_loss_pct': round(sl_pct * 100, 2),
                'stop_loss_price': round(entry_price * (1 - sl_pct), 8 if entry_price < 0.01 else 2),
                'take_profit_pct': round(tp_pct * 100, 2),
                'take_profit_price': round(entry_price * (1 + tp_pct), 8 if entry_price < 0.01 else 2),
                'max_loss_usd': round(pos_usd * sl_pct, 2),
                'max_hold_hours': trade_config['max_hold_hours']
            }
        
        # ============================================
        # FORECAST
        # ============================================
        moves = {
            'BTC_USDT': {'up': 5, 'down': 4},
            'ETH_USDT': {'up': 7, 'down': 5},
            'SOL_USDT': {'up': 10, 'down': 8},
            'PEPE_USDT': {'up': 15, 'down': 12}
        }.get(coin, {'up': 5, 'down': 4})
        
        sideways = max(0, 1 - win_prob - loss_prob)
        
        if win_prob > loss_prob and win_prob > sideways:
            direction = 'BULLISH'
        elif loss_prob > win_prob and loss_prob > sideways:
            direction = 'BEARISH'
        else:
            direction = 'SIDEWAYS'
        
        forecast = {
            'direction': direction,
            'current_price': current_price,
            'bull_target': round(current_price * (1 + moves['up']/100), 8 if current_price < 0.01 else 2),
            'bear_target': round(current_price * (1 - moves['down']/100), 8 if current_price < 0.01 else 2),
            'probabilities': {
                'up': round(win_prob * 100, 1),
                'sideways': round(sideways * 100, 1),
                'down': round(loss_prob * 100, 1)
            }
        }
        
        # ============================================
        # SUGGESTED ACTION (with new metrics)
        # ============================================
        suggested_action = self._get_suggested_action(
            verdict, win_prob, adjusted_threshold, regime, expectancy, readiness
        )
        
        # ============================================
        # FINAL RESPONSE
        # ============================================
        return {
            'coin': coin,
            'timestamp': datetime.now().isoformat(),
            'price': current_price,
            'price_source': 'WEBSOCKET' if ws_price else ('LIVE' if live_price else 'CACHED'),
            
            'capital': request.capital,
            'trade_type': request.trade_type.value,
            'experience_level': request.experience_level.value,
            
            'verdict': verdict,
            'confidence': confidence,
            
            'model_ran': True,
            
            # Probabilities
            'win_probability': round(win_prob * 100, 1),
            'loss_probability': round(loss_prob * 100, 1),
            'win_threshold_used': round(adjusted_threshold * 100, 1),
            
            # NEW: Clear, non-confusing metrics
            'expectancy': round(expectancy, 1),
            'expectancy_status': expectancy_status,
            
            'readiness': round(readiness, 1),
            'readiness_status': readiness_status,
            
            # Keep risk-adjusted EV for advanced users
            'risk_adjusted_ev': round(risk_adjusted_ev, 2),
            
            # Reasoning
            'reasoning': reasoning,
            'warnings': warnings,
            
            # Scenarios
            'active_scenarios': [s.dict() for s in scenario_result.active_scenarios],
            'scenario_count': len(scenario_result.active_scenarios),
            
            'forecast': forecast,
            'risk': risk,
            'suggested_action': suggested_action,
            
            'market_context': {
                'btc': btc_context,
                'regime': regime
            }
        }
    
    def _get_suggested_action(self, verdict, win_prob, threshold, regime, expectancy, readiness) -> dict:
        if verdict == 'BUY':
            return {
                'action': 'EXECUTE',
                'message': 'Conditions favorable – proceed with entry',
                'next_check': None,
                'why': f'Expectancy: +{expectancy:.1f}%, Readiness: +{readiness:.1f}%'
            }
        
        elif verdict == 'AVOID':
            return {
                'action': 'STAY_OUT',
                'message': 'High risk – do not enter',
                'next_check': '4H',
                'conditions': [
                    f'Need expectancy > 0% (currently {expectancy:.1f}%)',
                    'Wait for LOSS prob < 45%'
                ]
            }
        
        else:  # WAIT
            conditions = []
            
            if expectancy < 0:
                conditions.append(f'Need expectancy > 0% (currently {expectancy:.1f}%)')
            
            if readiness < 0:
                conditions.append(f'Need WIN > {threshold*100:.0f}% (currently {win_prob*100:.1f}%)')
            
            if regime.get('adx', 0) < 18:
                conditions.append(f'ADX needs > 18 (currently {regime.get("adx", 0)})')
            
            return {
                'action': 'MONITOR',
                'message': 'Not ready yet – monitor for improvement',
                'next_check': '1H' if readiness > -5 else '4H',
                'conditions': conditions if conditions else ['Wait for better setup']
            }
    
    def scan_all(self) -> dict:
        """Quick scan all coins with NEW metrics"""
        signals = []
        btc = self.market_context.get_btc_context()
        
        for coin in self.coins:
            try:
                req = AnalyzeRequest(coin=coin)
                result = self.analyze(req)
                
                # Determine signal_strength based on verdict and confidence
                if result['verdict'] == 'BUY':
                    signal_strength = 'STRONG' if result['confidence'] == 'HIGH' else 'MODERATE'
                elif result['verdict'] == 'AVOID':
                    signal_strength = 'DANGER'
                elif result['verdict'] == 'BLOCKED':
                    signal_strength = 'DANGER'
                else:
                    signal_strength = 'NEUTRAL' if result['confidence'] == 'LOW' else 'WEAK'

                signals.append({
                    'coin': coin,
                    'price': result['price'],
                    'price_source': result.get('price_source', 'LIVE'),
                    'verdict': result['verdict'],
                    'confidence': result['confidence'],
                    'signal_strength': signal_strength,
                    'win_probability': result.get('win_probability'),
                    'loss_probability': result.get('loss_probability'),
                    'warnings_count': len(result.get('warnings', [])),
                    # NEW: Use expectancy and readiness instead of edge
                    'expectancy': result.get('expectancy'),
                    'readiness': result.get('readiness'),
                    'scenario_count': result['scenario_count'],
                    'model_ran': result['model_ran']
                })
            except Exception as e:
                print(f"Scan error {coin}: {e}")
        
        # Sort by expectancy (most important for profitability)
        signals = sorted(signals, key=lambda x: x.get('expectancy') or -100, reverse=True)
        
        buy_signals = [s for s in signals if s['verdict'] == 'BUY']
        blocked = [s for s in signals if s['verdict'] == 'BLOCKED']
        
        if blocked:
            summary = f"⛔ {len(blocked)} coin(s) blocked by scenarios"
        elif buy_signals:
            summary = f"🟢 {len(buy_signals)} trade opportunity(s)!"
        elif any(s['verdict'] == 'AVOID' for s in signals):
            summary = "🔴 High risk – avoid trading"
        else:
            summary = "⏳ No opportunities – patience is profitable"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'market_context': {
                'btc': btc,
                'overall_risk': 'HIGH' if any(s['verdict'] in ['BLOCKED', 'AVOID'] for s in signals) else 'NORMAL'
            },
            'signals': signals,
            'buy_signals': buy_signals,
            'blocked_count': len(blocked),
            'market_summary': summary
        }


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Crypto AI Trading API",
    description="Professional trading system with Expectancy + Readiness metrics",
    version="4.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "="*60)
print("  🚀 Initializing Trading Engine v4.1...")
print("="*60)
engine = TradingEngine()
print("="*60 + "\n")


# ============================================================
# REST ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "Crypto AI Trading API",
        "version": "4.1.0",
        "features": [
            "Scenario Engine (gate layer)",
            "WebSocket live prices",
            "Decision Engine (ML)",
            "Risk Engine",
            "NEW: Expectancy (WIN% - LOSS%)",
            "NEW: Readiness (WIN% - Threshold%)"
        ],
        "websocket": "ws://localhost:8000/ws/prices"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": list(engine.models.keys()),
        "scenario_engine": "active",
        "price_streamer": "active" if price_streamer.running else "starting",
        "connected_clients": len(price_streamer.clients),
        "version": "4.1.0"
    }

@app.get("/scan")
async def scan():
    return engine.scan_all()

@app.post("/analyze")
async def analyze_post(request: AnalyzeRequest):
    try:
        return engine.analyze(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analyze/{coin}")
async def analyze_get(
    coin: str,
    capital: float = 1000,
    trade_type: TradeType = TradeType.SWING,
    experience: ExperienceLevel = ExperienceLevel.INTERMEDIATE,
    reason: Optional[TradeReason] = None,
    recent_losses: int = 0,
    trades_today: int = 0
):
    request = AnalyzeRequest(
        coin=coin,
        capital=capital,
        trade_type=trade_type,
        experience_level=experience,
        reason_for_trade=reason,
        recent_losses=recent_losses,
        trades_today=trades_today
    )
    return engine.analyze(request)

@app.get("/scenarios")
async def get_scenarios():
    return {
        "active_news_flags": engine.scenario_engine.get_active_news_flags(),
        "available_scenarios": [
            "MARKET_CRASH", "MARKET_STRESS", "WEEKEND", 
            "EXTENDED_MOVE", "FOMO_DETECTED", "OVERTRADING", "LOSING_STREAK"
        ]
    }

@app.post("/scenarios/news/{flag}")
async def set_news_flag(flag: str):
    engine.scenario_engine.set_news_flag(flag)
    return {"status": "set", "flag": flag}

@app.delete("/scenarios/news/{flag}")
async def clear_news_flag(flag: str):
    engine.scenario_engine.clear_news_flag(flag)
    return {"status": "cleared", "flag": flag}

@app.get("/scenarios/news/set/{flag}")
async def set_news_flag_get(flag: str):
    engine.scenario_engine.set_news_flag(flag)
    return {"status": "set", "flag": flag, "active_flags": engine.scenario_engine.get_active_news_flags()}

@app.get("/scenarios/news/clear/{flag}")
async def clear_news_flag_get(flag: str):
    engine.scenario_engine.clear_news_flag(flag)
    return {"status": "cleared", "flag": flag, "active_flags": engine.scenario_engine.get_active_news_flags()}


# ============================================================
# WEBSOCKET ENDPOINTS
# ============================================================

@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    await price_streamer.connect_client(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
            elif data == "get_prices":
                await websocket.send_json({
                    "type": "all_prices",
                    "prices": price_streamer.prices
                })
    except WebSocketDisconnect:
        price_streamer.disconnect_client(websocket)


@app.get("/prices/live")
async def get_live_prices():
    return {
        "timestamp": datetime.now().isoformat(),
        "prices": price_streamer.prices,
        "source": "binance_websocket",
        "connected_clients": len(price_streamer.clients)
    }


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(price_streamer.start())
    print("  🚀 Price streamer started")


@app.on_event("shutdown")
async def shutdown_event():
    price_streamer.stop()
    print("  🛑 Price streamer stopped")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "#"*60)
    print("  🚀 CRYPTO AI TRADING API v4.1")
    print("#"*60)
    print("\n  FIXED in v4.1:")
    print("    📊 Expectancy: WIN% - LOSS%")
    print("       → Is this trade profitable?")
    print("    📊 Readiness: WIN% - Threshold%")
    print("       → How close to entry?")
    print("\n  No more confusing 'Edge' metric!")
    print("\n  Server: http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("\n" + "#"*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)