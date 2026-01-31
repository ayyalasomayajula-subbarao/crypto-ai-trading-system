"""
Crypto AI Trading API - v5.0
Complete Trade-Type-Specific Analysis System

FEATURES in v5.0:
- Trade-type-specific thresholds (Scalp/Short/Swing/Investment)
- Experience level modifiers
- Enhanced FOMO detection with trade-type sensitivity
- ADX/Regime requirements per trade type
- Complete position sizing based on trade type
- Educational reasoning for users
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Set
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sanitize_for_json(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

# ============================================================
# TRADE TYPE CONFIGURATION
# ============================================================

TRADE_TYPE_CONFIG = {
    'SCALP': {
        'name': 'Scalp',
        'duration_hours': 4,
        'duration_display': 'Minutes to hours',
        'description': 'Quick trades capturing small price movements. Requires high win rate and strict discipline.',
        'risk_level': 'Very High',
        
        # Thresholds (stricter for scalping)
        'base_win_threshold': 0.55,
        'min_expectancy': 5.0,
        'max_loss_probability': 40.0,
        
        # Risk Management
        'position_size_pct': 0.08,
        'stop_loss_pct': 1.5,
        'take_profit_pct': 2.5,
        'min_risk_reward': 1.5,
        
        # Market Requirements
        'min_adx': 25,
        'allowed_regimes': ['TRENDING'],
        'blocked_volatility': ['EXTREME'],
        
        # Behavioral
        'fomo_sensitivity': 1.5,
        'max_trades_per_day': 10,
        'losing_streak_block': 2,
        
        # Time restrictions
        'blocked_hours': [0, 1, 2, 3, 4, 5],
        'weekend_allowed': False
    },
    
    'SHORT_TERM': {
        'name': 'Short Term',
        'duration_hours': 48,
        'duration_display': '1-2 days',
        'description': 'Capturing intraday swings. Balance between frequency and quality of setups.',
        'risk_level': 'High',
        
        'base_win_threshold': 0.50,
        'min_expectancy': 3.0,
        'max_loss_probability': 45.0,
        
        'position_size_pct': 0.12,
        'stop_loss_pct': 2.5,
        'take_profit_pct': 5.0,
        'min_risk_reward': 2.0,
        
        'min_adx': 20,
        'allowed_regimes': ['TRENDING', 'TRANSITIONING'],
        'blocked_volatility': ['EXTREME'],
        
        'fomo_sensitivity': 1.3,
        'max_trades_per_day': 6,
        'losing_streak_block': 2,
        
        'blocked_hours': [],
        'weekend_allowed': True
    },
    
    'SWING': {
        'name': 'Swing',
        'duration_hours': 168,
        'duration_display': '2-7 days',
        'description': 'Riding medium-term trends. Most balanced approach for beginners and intermediates.',
        'risk_level': 'Medium',
        
        'base_win_threshold': 0.45,
        'min_expectancy': 0.0,
        'max_loss_probability': 50.0,
        
        'position_size_pct': 0.15,
        'stop_loss_pct': 4.0,
        'take_profit_pct': 10.0,
        'min_risk_reward': 2.5,
        
        'min_adx': 15,
        'allowed_regimes': ['TRENDING', 'TRANSITIONING', 'RANGING'],
        'blocked_volatility': [],
        
        'fomo_sensitivity': 1.0,
        'max_trades_per_day': 4,
        'losing_streak_block': 3,
        
        'blocked_hours': [],
        'weekend_allowed': True
    },
    
    'INVESTMENT': {
        'name': 'Investment',
        'duration_hours': 720,
        'duration_display': 'Weeks to months',
        'description': 'Long-term position building. Lower win rate acceptable with larger targets.',
        'risk_level': 'Lower',
        
        'base_win_threshold': 0.35,
        'min_expectancy': -5.0,
        'max_loss_probability': 55.0,
        
        'position_size_pct': 0.25,
        'stop_loss_pct': 8.0,
        'take_profit_pct': 25.0,
        'min_risk_reward': 3.0,
        
        'min_adx': 0,
        'allowed_regimes': ['TRENDING', 'TRANSITIONING', 'RANGING', 'UNKNOWN'],
        'blocked_volatility': [],
        
        'fomo_sensitivity': 0.5,
        'max_trades_per_day': 2,
        'losing_streak_block': 4,
        
        'blocked_hours': [],
        'weekend_allowed': True
    }
}

EXPERIENCE_MODIFIERS = {
    'BEGINNER': {
        'name': 'Beginner',
        'threshold_boost': 0.10,
        'position_size_mult': 0.5,
        'fomo_sensitivity_mult': 1.5,
        'max_trades_mult': 0.5,
        'blocked_trade_types': ['SCALP'],
        'description': 'New to trading. Extra protection enabled.'
    },
    'INTERMEDIATE': {
        'name': 'Intermediate',
        'threshold_boost': 0.05,
        'position_size_mult': 0.75,
        'fomo_sensitivity_mult': 1.2,
        'max_trades_mult': 0.75,
        'blocked_trade_types': [],
        'description': 'Some experience. Moderate protection.'
    },
    'ADVANCED': {
        'name': 'Advanced',
        'threshold_boost': 0.0,
        'position_size_mult': 1.0,
        'fomo_sensitivity_mult': 1.0,
        'max_trades_mult': 1.0,
        'blocked_trade_types': [],
        'description': 'Experienced trader. Full access.'
    }
}

# ============================================================
# PYDANTIC MODELS
# ============================================================

class TradeType(str, Enum):
    SCALP = "SCALP"
    SHORT_TERM = "SHORT_TERM"
    SWING = "SWING"
    INVESTMENT = "INVESTMENT"

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

class AnalyzeRequest(BaseModel):
    coin: str
    capital: float = 1000
    trade_type: TradeType = TradeType.SWING
    experience: ExperienceLevel = ExperienceLevel.INTERMEDIATE
    reason: Optional[TradeReason] = None
    recent_losses: int = 0
    trades_today: int = 0
    entry_price: Optional[float] = None

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Crypto AI Trading API",
    version="5.0.0",
    description="Trade-Type-Specific Analysis System"
)

# CORS - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Exception handler to ensure CORS headers on errors
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"❌ Global error: {exc}")
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

# ============================================================
# PRICE STREAMER (WebSocket to Binance)
# ============================================================

class PriceStreamer:
    def __init__(self):
        self.prices: Dict[str, Dict] = {}
        self.clients: Set[WebSocket] = set()
        self.running = False
        
    async def connect_binance(self, symbol: str):
        """Connect to Binance WebSocket for a single symbol"""
        import websockets
        
        ws_symbol = symbol.replace('_', '').lower()
        url = f"wss://stream.binance.com:9443/ws/{ws_symbol}@ticker"
        
        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    print(f"✅ Connected to Binance: {symbol}")
                    async for msg in ws:
                        if not self.running:
                            break
                        data = json.loads(msg)
                        
                        price_data = {
                            'price': float(data['c']),
                            'change_24h': float(data['P']),
                            'high_24h': float(data['h']),
                            'low_24h': float(data['l']),
                            'volume_24h': float(data['v']),
                            'direction': 'up' if float(data['p']) > 0 else 'down' if float(data['p']) < 0 else 'same',
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        old_price = self.prices.get(symbol, {}).get('price', 0)
                        new_price = price_data['price']
                        if old_price > 0:
                            price_data['direction'] = 'up' if new_price > old_price else 'down' if new_price < old_price else 'same'
                        
                        self.prices[symbol] = price_data
                        await self.broadcast(symbol, price_data)
                        
            except Exception as e:
                print(f"❌ Binance connection error ({symbol}): {e}")
                await asyncio.sleep(5)
    
    async def broadcast(self, symbol: str, data: Dict):
        """Broadcast price update to all connected clients"""
        if not self.clients:
            return
            
        message = json.dumps({
            'type': 'price_update',
            'coin': symbol,
            'data': data
        })
        
        disconnected = set()
        for client in self.clients:
            try:
                await client.send_text(message)
            except:
                disconnected.add(client)
        
        self.clients -= disconnected
    
    async def start(self, symbols: List[str]):
        """Start streaming for all symbols"""
        self.running = True
        tasks = [self.connect_binance(s) for s in symbols]
        await asyncio.gather(*tasks)
    
    def stop(self):
        """Stop all streams"""
        self.running = False

# Global price streamer
price_streamer = PriceStreamer()

# ============================================================
# TRADING ENGINE
# ============================================================

class TradingEngine:
    def __init__(self):
        self.coins = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']
        self.models = {}
        self.data_cache = {}
        self.load_models()
    
    def load_models(self):
        """Load ML models for each coin"""
        model_dir = 'models'
        for coin in self.coins:
            # Try the correct path structure: models/{coin}/decision_model.pkl
            model_path = os.path.join(model_dir, coin, 'decision_model.pkl')
            if os.path.exists(model_path):
                try:
                    self.models[coin] = joblib.load(model_path)
                    # Also load the feature list
                    feature_list_path = os.path.join(model_dir, coin, 'feature_list.txt')
                    if os.path.exists(feature_list_path):
                        with open(feature_list_path, 'r') as f:
                            self.feature_lists = getattr(self, 'feature_lists', {})
                            self.feature_lists[coin] = [line.strip() for line in f if line.strip()]
                    print(f"✅ Loaded model: {coin}")
                except Exception as e:
                    print(f"❌ Error loading {coin} model: {e}")
            else:
                # Fallback to old path structure
                old_model_path = os.path.join(model_dir, f'{coin}_model.pkl')
                if os.path.exists(old_model_path):
                    try:
                        self.models[coin] = joblib.load(old_model_path)
                        print(f"✅ Loaded model (legacy): {coin}")
                    except Exception as e:
                        print(f"❌ Error loading {coin} model: {e}")
    
    def get_live_price(self, coin: str) -> tuple:
        """Get live price from streamer or fallback to API"""
        if coin in price_streamer.prices:
            data = price_streamer.prices[coin]
            return data['price'], 'WEBSOCKET'
        
        # Fallback to REST API
        try:
            import ccxt
            exchange = ccxt.binance()
            symbol = coin.replace('_', '/')
            ticker = exchange.fetch_ticker(symbol)
            return ticker['last'], 'REST_API'
        except:
            return 0, 'UNAVAILABLE'
    
    def load_data(self, coin: str, multi_tf: bool = False) -> Optional[pd.DataFrame]:
        """Load historical data for a coin

        Args:
            coin: Coin symbol (e.g., 'BTC_USDT')
            multi_tf: If True, load multi-timeframe features file for model prediction
        """
        if multi_tf:
            data_path = f'data/{coin}_multi_tf_features.csv'
        else:
            data_path = f'data/{coin}_1h.csv'

        if os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                print(f"Error loading data {data_path}: {e}")
                pass
        return None
    
    def get_btc_context(self) -> Dict:
        """Get BTC market context"""
        btc_data = self.load_data('BTC_USDT')
        price, source = self.get_live_price('BTC_USDT')
        
        context = {
            'price': price,
            'price_source': source,
            'change_24h': price_streamer.prices.get('BTC_USDT', {}).get('change_24h', 0),
            'trend_1h': 'SIDEWAYS',
            'trend_4h': 'SIDEWAYS',
            'trend_1d': 'SIDEWAYS',
            'overall_trend': 'NEUTRAL',
            'support_alts': True
        }
        
        if btc_data is not None and len(btc_data) >= 24:
            close = btc_data['close'].values
            
            # 1H trend
            if len(close) >= 2:
                change_1h = (close[-1] - close[-2]) / close[-2] * 100
                context['trend_1h'] = 'UP' if change_1h > 0.5 else 'DOWN' if change_1h < -0.5 else 'SIDEWAYS'
            
            # 4H trend
            if len(close) >= 5:
                change_4h = (close[-1] - close[-5]) / close[-5] * 100
                context['trend_4h'] = 'UP' if change_4h > 1 else 'DOWN' if change_4h < -1 else 'SIDEWAYS'
            
            # 1D trend
            if len(close) >= 24:
                change_1d = (close[-1] - close[-24]) / close[-24] * 100
                context['trend_1d'] = 'UP' if change_1d > 2 else 'DOWN' if change_1d < -2 else 'SIDEWAYS'
            
            # Overall
            trends = [context['trend_1h'], context['trend_4h'], context['trend_1d']]
            up_count = trends.count('UP')
            down_count = trends.count('DOWN')
            
            if up_count >= 2:
                context['overall_trend'] = 'BULLISH'
            elif down_count >= 2:
                context['overall_trend'] = 'BEARISH'
            else:
                context['overall_trend'] = 'NEUTRAL'
            
            context['support_alts'] = context['overall_trend'] != 'BEARISH'
        
        return context
    
    def get_market_regime(self, coin: str) -> Dict:
        """Detect market regime using ADX"""
        result = {
            'regime': 'UNKNOWN',
            'adx': 0,
            'volatility': 'UNKNOWN',
            'volatility_pct': 0,
            'recommendation': ''
        }

        # First try to get pre-computed ADX from multi-tf features
        df_features = self.load_data(coin, multi_tf=True)
        if df_features is not None and len(df_features) > 0 and '1h_adx' in df_features.columns:
            try:
                adx = df_features['1h_adx'].iloc[-1]
                if pd.notna(adx):
                    result['adx'] = round(float(adx), 1)

                    # Get volatility from ATR percentage if available
                    if '1h_atr_pct' in df_features.columns:
                        atr_pct = df_features['1h_atr_pct'].iloc[-1]
                        if pd.notna(atr_pct):
                            result['volatility_pct'] = round(float(atr_pct), 2)
            except Exception as e:
                print(f"Error reading pre-computed ADX: {e}")

        # Fallback to manual calculation if needed
        if result['adx'] == 0:
            df = self.load_data(coin)
            if df is None or len(df) < 20:
                result['recommendation'] = 'Insufficient data for regime detection'
                return result

            try:
                high = df['high'].values
                low = df['low'].values
                close = df['close'].values

                # Calculate ADX
                tr = np.maximum(high[1:] - low[1:],
                              np.abs(high[1:] - close[:-1]),
                              np.abs(low[1:] - close[:-1]))
                atr = pd.Series(tr).rolling(14).mean().iloc[-1]

                # Simplified ADX calculation
                plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                                  np.maximum(high[1:] - high[:-1], 0), 0)
                minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                                   np.maximum(low[:-1] - low[1:], 0), 0)

                plus_di = 100 * pd.Series(plus_dm).rolling(14).mean().iloc[-1] / atr if atr > 0 else 0
                minus_di = 100 * pd.Series(minus_dm).rolling(14).mean().iloc[-1] / atr if atr > 0 else 0

                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
                adx = pd.Series([dx] * 14).rolling(14).mean().iloc[-1] if not np.isnan(dx) else 0

                result['adx'] = round(adx, 1)
            except Exception as e:
                print(f"ADX calculation error: {e}")
                result['adx'] = 0

        # Now determine regime based on ADX
        adx = result['adx']

        # Determine regime
        if adx >= 40:
            result['regime'] = 'TRENDING'
            result['recommendation'] = 'Strong trend - good for trend-following strategies'
        elif adx >= 25:
            result['regime'] = 'TRENDING'
            result['recommendation'] = 'Moderate trend - proceed with directional bias'
        elif adx >= 20:
            result['regime'] = 'TRANSITIONING'
            result['recommendation'] = 'Trend developing - wait for confirmation'
        else:
            result['regime'] = 'RANGING'
            result['recommendation'] = 'No clear trend - consider range strategies or wait'

        # Calculate volatility if not already set from pre-computed data
        if result['volatility_pct'] == 0:
            try:
                df = self.load_data(coin)
                if df is not None and len(df) >= 24 and 'close' in df.columns:
                    close = df['close'].values
                    returns = pd.Series(close).pct_change().dropna()
                    vol = returns.std() * np.sqrt(24) * 100  # Annualized hourly vol
                    result['volatility_pct'] = round(vol, 2)
            except Exception as e:
                print(f"Volatility calculation error: {e}")

        # Determine volatility level
        vol = result['volatility_pct']
        if vol > 100:
            result['volatility'] = 'EXTREME'
        elif vol > 60:
            result['volatility'] = 'HIGH'
        elif vol > 30:
            result['volatility'] = 'MODERATE'
        elif vol > 0:
            result['volatility'] = 'LOW'
        else:
            result['volatility'] = 'UNKNOWN'

        return result
    
    def get_model_probabilities(self, coin: str) -> Dict:
        """Get WIN/LOSS/SIDEWAYS probabilities from model"""
        if coin not in self.models:
            return {'win': 0, 'loss': 0, 'sideways': 0, 'error': 'Model not loaded'}

        # Load multi-timeframe features
        df = self.load_data(coin, multi_tf=True)
        if df is None or len(df) < 50:
            return {'win': 0, 'loss': 0, 'sideways': 0, 'error': 'Insufficient data'}

        try:
            model = self.models[coin]

            # Get the feature list for this coin
            feature_lists = getattr(self, 'feature_lists', {})
            feature_list = feature_lists.get(coin, [])

            if not feature_list:
                # Fallback: try to infer features from dataframe columns
                # Exclude non-feature columns
                exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                               'target_return', 'target_direction']
                feature_list = [col for col in df.columns if col not in exclude_cols]

            if not feature_list:
                return {'win': 0, 'loss': 0, 'sideways': 0, 'error': 'No features available'}

            # Get the latest row of features
            latest_row = df.iloc[-1]

            # Extract features in the correct order
            features = []
            for feat in feature_list:
                if feat in latest_row:
                    val = latest_row[feat]
                    # Handle NaN values
                    if pd.isna(val):
                        val = 0.0
                    features.append(float(val))
                else:
                    features.append(0.0)

            # Predict - use DataFrame with feature names to avoid sklearn warning
            X = pd.DataFrame([features], columns=feature_list)

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[0]
                classes = model.classes_ if hasattr(model, 'classes_') else None

                # Map probabilities based on class labels
                if classes is not None:
                    class_list = list(classes)
                    win_prob = 0.0
                    loss_prob = 0.0
                    sideways_prob = 0.0

                    for i, cls in enumerate(class_list):
                        cls_str = str(cls).upper()
                        if cls_str in ['UP', '1', 'WIN', 'BULLISH']:
                            win_prob = probs[i] * 100
                        elif cls_str in ['DOWN', '-1', '0', 'LOSS', 'BEARISH']:
                            loss_prob = probs[i] * 100
                        elif cls_str in ['SIDEWAYS', 'NEUTRAL']:
                            sideways_prob = probs[i] * 100

                    # If we couldn't map classes, use position-based assignment
                    if win_prob == 0 and loss_prob == 0:
                        if len(probs) >= 3:
                            loss_prob = probs[0] * 100
                            sideways_prob = probs[1] * 100
                            win_prob = probs[2] * 100
                        elif len(probs) == 2:
                            loss_prob = probs[0] * 100
                            win_prob = probs[1] * 100
                else:
                    # Fallback for models without classes_ attribute
                    if len(probs) >= 3:
                        loss_prob = probs[0] * 100
                        sideways_prob = probs[1] * 100
                        win_prob = probs[2] * 100
                    elif len(probs) == 2:
                        loss_prob = probs[0] * 100
                        win_prob = probs[1] * 100
                    else:
                        win_prob = probs[0] * 100
                        loss_prob = 100 - win_prob
                        sideways_prob = 0.0

                return {
                    'win': round(win_prob, 1),
                    'loss': round(loss_prob, 1),
                    'sideways': round(sideways_prob, 1),
                    'error': None
                }
            else:
                # Binary model without predict_proba
                pred = model.predict(X)[0]
                pred_str = str(pred).upper()
                if pred_str in ['UP', '1', 'WIN', 'BULLISH']:
                    return {'win': 60, 'loss': 40, 'sideways': 0, 'error': None}
                else:
                    return {'win': 40, 'loss': 60, 'sideways': 0, 'error': None}

        except Exception as e:
            import traceback
            error_msg = f"Model prediction error for {coin}: {e}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            # Return fallback values instead of crashing
            return {'win': 45, 'loss': 45, 'sideways': 10, 'error': error_msg}
    
    def apply_trade_type_analysis(
        self,
        trade_type: str,
        experience: str,
        win_prob: float,
        loss_prob: float,
        adx: float,
        regime: str,
        volatility: str,
        user_context: Dict
    ) -> Dict:
        """Apply trade-type-specific analysis rules"""
        
        config = TRADE_TYPE_CONFIG.get(trade_type, TRADE_TYPE_CONFIG['SWING'])
        exp_mod = EXPERIENCE_MODIFIERS.get(experience, EXPERIENCE_MODIFIERS['INTERMEDIATE'])
        
        result = {
            'adjusted_threshold': config['base_win_threshold'],
            'position_size_pct': config['position_size_pct'],
            'stop_loss_pct': config['stop_loss_pct'],
            'take_profit_pct': config['take_profit_pct'],
            'max_hold_hours': config['duration_hours'],
            'warnings': [],
            'blocks': [],
            'reasoning': [],
            'trade_type_requirements': []
        }
        
        # =====================================
        # 1. EXPERIENCE MODIFIERS
        # =====================================
        
        if trade_type in exp_mod['blocked_trade_types']:
            result['blocks'].append({
                'type': 'EXPERIENCE_BLOCK',
                'message': f'{config["name"]} trading not recommended for {exp_mod["name"]} traders',
                'severity': 'HIGH'
            })
            result['reasoning'].append(f'{config["name"]} requires more trading experience')
        
        result['adjusted_threshold'] += exp_mod['threshold_boost']
        
        if exp_mod['threshold_boost'] > 0:
            result['reasoning'].append(
                f'WIN threshold: {config["base_win_threshold"]*100:.0f}% base + '
                f'{exp_mod["threshold_boost"]*100:.0f}% ({exp_mod["name"]}) = '
                f'{result["adjusted_threshold"]*100:.0f}%'
            )
        else:
            result['reasoning'].append(
                f'WIN threshold: {result["adjusted_threshold"]*100:.0f}% for {config["name"]} trading'
            )
        
        result['position_size_pct'] *= exp_mod['position_size_mult']
        
        # =====================================
        # 2. MARKET REGIME CHECK
        # =====================================
        
        if regime not in config['allowed_regimes'] and regime != 'UNKNOWN':
            result['warnings'].append({
                'type': 'REGIME_MISMATCH',
                'message': f'{regime} market not ideal for {config["name"]} trading',
                'severity': 'MEDIUM'
            })
            result['adjusted_threshold'] += 0.05
            result['reasoning'].append(f'+5% threshold: {regime} regime not optimal for {config["name"]}')
        
        result['trade_type_requirements'].append({
            'name': 'Market Regime',
            'required': ', '.join(config['allowed_regimes']),
            'current': regime,
            'met': bool(regime in config['allowed_regimes'] or regime == 'UNKNOWN')
        })
        
        # =====================================
        # 3. ADX CHECK
        # =====================================
        
        if config['min_adx'] > 0:
            if adx < config['min_adx']:
                if trade_type == 'SCALP':
                    result['blocks'].append({
                        'type': 'ADX_BLOCK',
                        'message': f'Scalping requires ADX > {config["min_adx"]} (current: {adx:.0f})',
                        'severity': 'HIGH'
                    })
                else:
                    result['warnings'].append({
                        'type': 'WEAK_TREND',
                        'message': f'ADX {adx:.0f} below {config["min_adx"]} recommended for {config["name"]}',
                        'severity': 'LOW'
                    })
                result['reasoning'].append(f'Weak trend: ADX {adx:.0f} < {config["min_adx"]} required')
            
            result['trade_type_requirements'].append({
                'name': 'Trend Strength (ADX)',
                'required': f'> {config["min_adx"]}',
                'current': f'{float(adx):.0f}',
                'met': bool(adx >= config['min_adx'])
            })
        
        # =====================================
        # 4. VOLATILITY CHECK
        # =====================================
        
        if volatility in config['blocked_volatility']:
            result['blocks'].append({
                'type': 'VOLATILITY_BLOCK',
                'message': f'{volatility} volatility too dangerous for {config["name"]} trading',
                'severity': 'HIGH'
            })
        
        # =====================================
        # 5. TIME RESTRICTIONS
        # =====================================
        
        current_hour = datetime.now().hour
        is_weekend = datetime.now().weekday() >= 5
        
        if current_hour in config['blocked_hours']:
            result['warnings'].append({
                'type': 'TIME_WARNING',
                'message': f'{config["name"]} not recommended during off-hours (current: {current_hour}:00)',
                'severity': 'MEDIUM'
            })
            result['adjusted_threshold'] += 0.05
        
        if is_weekend and not config['weekend_allowed']:
            result['blocks'].append({
                'type': 'WEEKEND_BLOCK',
                'message': f'{config["name"]} trading blocked on weekends',
                'severity': 'HIGH'
            })
        
        # =====================================
        # 6. PROBABILITY REQUIREMENTS
        # =====================================
        
        if loss_prob > config['max_loss_probability']:
            result['warnings'].append({
                'type': 'HIGH_LOSS_PROB',
                'message': f'LOSS {loss_prob:.1f}% exceeds {config["name"]} max of {config["max_loss_probability"]}%',
                'severity': 'HIGH'
            })
            result['reasoning'].append(
                f'High loss risk: {loss_prob:.1f}% > {config["max_loss_probability"]}% limit'
            )
        
        result['trade_type_requirements'].append({
            'name': 'Max Loss Probability',
            'required': f'< {config["max_loss_probability"]}%',
            'current': f'{loss_prob:.1f}%',
            'met': bool(loss_prob <= config['max_loss_probability'])
        })
        
        expectancy = win_prob - loss_prob
        if expectancy < config['min_expectancy']:
            result['warnings'].append({
                'type': 'LOW_EXPECTANCY',
                'message': f'Expectancy {expectancy:.1f}% below {config["name"]} minimum of {config["min_expectancy"]}%',
                'severity': 'MEDIUM' if expectancy > -5 else 'HIGH'
            })
            result['reasoning'].append(
                f'Low expectancy: {expectancy:.1f}% < {config["min_expectancy"]}% required'
            )
        
        result['trade_type_requirements'].append({
            'name': 'Minimum Expectancy',
            'required': f'> {config["min_expectancy"]}%',
            'current': f'{expectancy:.1f}%',
            'met': bool(expectancy >= config['min_expectancy'])
        })
        
        # =====================================
        # 7. BEHAVIORAL LIMITS
        # =====================================
        
        trades_today = user_context.get('trades_today', 0)
        max_trades = int(config['max_trades_per_day'] * exp_mod['max_trades_mult'])
        
        if trades_today >= max_trades:
            result['blocks'].append({
                'type': 'TRADE_LIMIT',
                'message': f'Daily limit reached: {trades_today}/{max_trades} trades',
                'severity': 'CRITICAL'
            })
        elif trades_today >= max_trades - 1:
            result['warnings'].append({
                'type': 'NEAR_LIMIT',
                'message': f'Approaching daily limit: {trades_today}/{max_trades} trades',
                'severity': 'MEDIUM'
            })
        
        recent_losses = user_context.get('recent_losses', 0)
        if recent_losses >= config['losing_streak_block']:
            result['blocks'].append({
                'type': 'LOSING_STREAK',
                'message': f'{recent_losses} consecutive losses - take a break',
                'severity': 'CRITICAL'
            })
        
        # =====================================
        # 8. FOMO CHECK
        # =====================================
        
        reason = user_context.get('reason', '')
        if reason == 'FOMO':
            fomo_boost = 0.10 * config['fomo_sensitivity'] * exp_mod['fomo_sensitivity_mult']
            result['adjusted_threshold'] += fomo_boost
            result['warnings'].append({
                'type': 'FOMO_DETECTED',
                'message': f'FOMO detected - threshold raised by {fomo_boost*100:.0f}%',
                'severity': 'HIGH'
            })
            result['reasoning'].append(
                f'+{fomo_boost*100:.0f}% threshold: FOMO protection for {config["name"]}'
            )
        
        if reason == 'TIP':
            result['warnings'].append({
                'type': 'TIP_WARNING',
                'message': 'Trade based on tip - verify with your own analysis',
                'severity': 'MEDIUM'
            })
            result['adjusted_threshold'] += 0.05
        
        if reason == 'DIP_BUY':
            result['warnings'].append({
                'type': 'DIP_BUY_WARNING',
                'message': 'Catching falling knives is risky - ensure support confirmed',
                'severity': 'LOW'
            })
        
        # =====================================
        # 9. RISK/REWARD
        # =====================================
        
        risk_reward = result['take_profit_pct'] / result['stop_loss_pct'] if result['stop_loss_pct'] > 0 else 0
        result['risk_reward_ratio'] = round(risk_reward, 2)
        
        result['trade_type_requirements'].append({
            'name': 'Risk/Reward Ratio',
            'required': f'> {config["min_risk_reward"]}:1',
            'current': f'{risk_reward:.1f}:1',
            'met': bool(risk_reward >= config['min_risk_reward'])
        })
        
        # =====================================
        # 10. THRESHOLD CAP
        # =====================================
        
        result['adjusted_threshold'] = min(result['adjusted_threshold'], 0.75)
        
        # Add trade type info
        result['trade_type_info'] = {
            'name': config['name'],
            'duration_hours': config['duration_hours'],
            'duration_display': config['duration_display'],
            'description': config['description'],
            'risk_level': config['risk_level'],
            'base_threshold': config['base_win_threshold'],
            'min_adx': config['min_adx'],
            'min_expectancy': config['min_expectancy'],
            'max_loss_prob': config['max_loss_probability']
        }
        
        result['experience_info'] = {
            'name': exp_mod['name'],
            'description': exp_mod['description'],
            'threshold_boost': exp_mod['threshold_boost'],
            'position_mult': exp_mod['position_size_mult']
        }
        
        return result
    
    def analyze(self, request: AnalyzeRequest) -> Dict:
        """Main analysis function with trade-type-specific logic"""
        
        coin = request.coin.upper()
        if '_' not in coin:
            coin = f"{coin}_USDT"
        
        # Get live price
        price, price_source = self.get_live_price(coin)
        
        # Get BTC context
        btc = self.get_btc_context()
        
        # Get market regime
        regime = self.get_market_regime(coin)
        
        # Get model probabilities
        probs = self.get_model_probabilities(coin)
        
        win_prob = probs.get('win', 0)
        loss_prob = probs.get('loss', 0)
        sideways_prob = probs.get('sideways', 0)
        
        # Apply trade-type-specific analysis
        trade_analysis = self.apply_trade_type_analysis(
            trade_type=request.trade_type.value,
            experience=request.experience.value,
            win_prob=win_prob,
            loss_prob=loss_prob,
            adx=regime.get('adx', 0),
            regime=regime.get('regime', 'UNKNOWN'),
            volatility=regime.get('volatility', 'UNKNOWN'),
            user_context={
                'trades_today': request.trades_today,
                'recent_losses': request.recent_losses,
                'reason': request.reason.value if request.reason else ''
            }
        )
        
        adjusted_threshold = trade_analysis['adjusted_threshold']
        reasoning = trade_analysis['reasoning'].copy()
        warnings = []
        
        # Convert warnings to list format
        for w in trade_analysis['warnings']:
            warnings.append(w['message'])
        
        # =====================================
        # DETERMINE VERDICT
        # =====================================
        
        verdict = 'WAIT'
        confidence = 'LOW'
        block_reason = None
        
        # Check for blocks first
        if trade_analysis['blocks']:
            verdict = 'BLOCKED'
            confidence = 'N/A'
            block_reason = trade_analysis['blocks'][0]['message']
            reasoning.insert(0, f"BLOCKED: {block_reason}")
        else:
            # Calculate metrics
            expectancy = win_prob - loss_prob
            readiness = win_prob - (adjusted_threshold * 100)
            
            # Determine verdict based on metrics
            if win_prob >= adjusted_threshold * 100 and expectancy >= 0:
                verdict = 'BUY'
                confidence = 'HIGH' if expectancy > 10 else 'MEDIUM' if expectancy > 5 else 'LOW'
                reasoning.append(f'WIN {win_prob:.1f}% meets threshold {adjusted_threshold*100:.0f}%')
                reasoning.append(f'Positive expectancy: +{expectancy:.1f}%')
                
            elif win_prob >= adjusted_threshold * 100 and expectancy < 0:
                verdict = 'WAIT'
                confidence = 'LOW'
                reasoning.append(f'Threshold met but expectancy negative ({expectancy:.1f}%)')
                reasoning.append('Wait for LOSS probability to decrease')
                
            elif loss_prob > 50:
                verdict = 'AVOID'
                confidence = 'HIGH'
                reasoning.append(f'LOSS {loss_prob:.1f}% > 50% - unfavorable odds')
                
            else:
                verdict = 'WAIT'
                confidence = 'MEDIUM' if readiness > -5 else 'LOW'
                reasoning.append(f'WIN {win_prob:.1f}% below threshold {adjusted_threshold*100:.0f}%')
                if readiness > -10:
                    reasoning.append(f'Close to entry - monitor for improvement')
        
        # =====================================
        # CALCULATE METRICS
        # =====================================
        
        expectancy = win_prob - loss_prob
        readiness = win_prob - (adjusted_threshold * 100)
        
        # Expectancy status
        if expectancy > 10:
            expectancy_status = '✅ Strong positive'
        elif expectancy > 5:
            expectancy_status = '✅ Positive'
        elif expectancy > 0:
            expectancy_status = '⚠️ Slightly positive'
        elif expectancy > -5:
            expectancy_status = '⚠️ Slightly negative'
        else:
            expectancy_status = '❌ Negative'
        
        # Readiness status
        if readiness > 10:
            readiness_status = '✅ Well above threshold'
        elif readiness >= 0:
            readiness_status = '✅ Ready for entry'
        elif readiness > -5:
            readiness_status = '⚠️ Near threshold'
        else:
            readiness_status = '❌ Below threshold'
        
        # =====================================
        # POSITION SIZING
        # =====================================
        
        if verdict == 'BUY':
            position_size_usd = round(request.capital * trade_analysis['position_size_pct'], 2)
            entry_price = request.entry_price or price
            stop_loss_price = round(entry_price * (1 - trade_analysis['stop_loss_pct'] / 100), 8)
            take_profit_price = round(entry_price * (1 + trade_analysis['take_profit_pct'] / 100), 8)
            max_loss = round(position_size_usd * trade_analysis['stop_loss_pct'] / 100, 2)
            
            risk_info = {
                'action': 'OPEN_POSITION',
                'position_size_usd': position_size_usd,
                'position_size_pct': round(trade_analysis['position_size_pct'] * 100, 1),
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'stop_loss_pct': trade_analysis['stop_loss_pct'],
                'take_profit_price': take_profit_price,
                'take_profit_pct': trade_analysis['take_profit_pct'],
                'max_loss_usd': max_loss,
                'max_hold_hours': trade_analysis['max_hold_hours'],
                'risk_reward': trade_analysis['risk_reward_ratio']
            }
        else:
            risk_info = {
                'action': 'NO_TRADE',
                'reason': block_reason if verdict == 'BLOCKED' else 'Conditions not met'
            }
        
        # =====================================
        # FORECAST
        # =====================================
        
        df = self.load_data(coin)
        if df is not None and len(df) >= 24:
            close = df['close'].values
            volatility = np.std(close[-24:]) / np.mean(close[-24:]) * 100
            bull_target = price * (1 + volatility * 1.5 / 100)
            bear_target = price * (1 - volatility * 1.5 / 100)
        else:
            bull_target = price * 1.15
            bear_target = price * 0.85
        
        direction = 'BULLISH' if win_prob > loss_prob else 'BEARISH' if loss_prob > win_prob else 'SIDEWAYS'
        
        forecast = {
            'direction': direction,
            'current_price': price,
            'bull_target': round(bull_target, 8),
            'bear_target': round(bear_target, 8),
            'probabilities': {
                'up': win_prob,
                'sideways': sideways_prob,
                'down': loss_prob
            }
        }
        
        # =====================================
        # SUGGESTED ACTION
        # =====================================
        
        if verdict == 'BUY':
            suggested_action = {
                'action': 'EXECUTE',
                'message': f'Conditions favorable for {trade_analysis["trade_type_info"]["name"]} trade',
                'next_check': None,
                'conditions': []
            }
        elif verdict == 'BLOCKED':
            suggested_action = {
                'action': 'STOP',
                'message': block_reason,
                'next_check': '1H' if 'FOMO' in (block_reason or '') else '4H',
                'conditions': [b['message'] for b in trade_analysis['blocks']]
            }
        elif verdict == 'AVOID':
            suggested_action = {
                'action': 'STAY_OUT',
                'message': 'Unfavorable conditions - do not enter',
                'next_check': '4H',
                'conditions': [
                    f'Need LOSS < 50% (currently {loss_prob:.1f}%)',
                    f'Need positive expectancy (currently {expectancy:.1f}%)'
                ]
            }
        else:
            conditions = []
            if expectancy < 0:
                conditions.append(f'Need expectancy > 0% (currently {expectancy:.1f}%)')
            if readiness < 0:
                conditions.append(f'Need WIN > {adjusted_threshold*100:.0f}% (currently {win_prob:.1f}%)')
            
            suggested_action = {
                'action': 'MONITOR',
                'message': 'Setup developing - not ready yet',
                'next_check': '1H' if readiness > -5 else '4H',
                'conditions': conditions
            }
        
        # =====================================
        # BUILD RESPONSE
        # =====================================

        response = {
            'coin': coin,
            'timestamp': datetime.now().isoformat(),
            'price': price,
            'price_source': price_source,
            
            # User inputs
            'capital': request.capital,
            'trade_type': request.trade_type.value,
            'experience_level': request.experience.value,
            'trade_reason': request.reason.value if request.reason else None,
            
            # Verdict
            'verdict': verdict,
            'confidence': confidence,
            'model_ran': probs.get('error') is None,
            'blocked_by': trade_analysis['blocks'][0]['type'] if trade_analysis['blocks'] else None,
            'block_reason': block_reason,
            
            # Probabilities
            'win_probability': win_prob,
            'loss_probability': loss_prob,
            'sideways_probability': sideways_prob,
            'win_threshold_used': round(adjusted_threshold * 100, 1),
            
            # Key metrics
            'expectancy': round(expectancy, 1),
            'expectancy_status': expectancy_status,
            'readiness': round(readiness, 1),
            'readiness_status': readiness_status,
            
            # Reasoning
            'reasoning': reasoning,
            'warnings': warnings,
            
            # Trade type analysis
            'trade_type_info': trade_analysis['trade_type_info'],
            'experience_info': trade_analysis['experience_info'],
            'trade_type_requirements': trade_analysis['trade_type_requirements'],
            
            # Risk
            'risk': risk_info,
            
            # Forecast
            'forecast': forecast,
            
            # Suggested action
            'suggested_action': suggested_action,
            
            # Market context
            'market_context': {
                'btc': btc,
                'regime': regime
            },
            
            # Scenarios (simplified)
            'active_scenarios': [
                {
                    'type': w.get('type', 'WARNING'),
                    'title': w.get('type', 'Warning').replace('_', ' ').title(),
                    'message': w.get('message', ''),
                    'icon': '⚠️',
                    'severity': w.get('severity', 'MEDIUM'),
                    'effect': 'MODIFY'
                }
                for w in trade_analysis['warnings']
            ],
            'scenario_count': len(trade_analysis['warnings']) + len(trade_analysis['blocks'])
        }

        # Sanitize numpy types for JSON serialization
        return sanitize_for_json(response)

    def scan_all(self) -> Dict:
        """Quick scan of all coins"""
        signals = []
        btc = self.get_btc_context()
        
        for coin in self.coins:
            try:
                request = AnalyzeRequest(coin=coin)
                result = self.analyze(request)
                
                signals.append({
                    'coin': coin,
                    'price': result['price'],
                    'price_source': result.get('price_source', 'CACHED'),
                    'verdict': result['verdict'],
                    'confidence': result['confidence'],
                    'win_probability': result.get('win_probability'),
                    'loss_probability': result.get('loss_probability'),
                    'expectancy': result.get('expectancy'),
                    'readiness': result.get('readiness'),
                    'forecast': {
                        'direction': result.get('forecast', {}).get('direction', 'UNKNOWN'),
                        'probabilities': result.get('forecast', {}).get('probabilities', {})
                    },
                    'scenario_count': result['scenario_count'],
                    'model_ran': result['model_ran']
                })
            except Exception as e:
                print(f"Scan error {coin}: {e}")
        
        # Sort by expectancy
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
            'market_context': {'btc': btc},
            'signals': signals,
            'buy_signals': buy_signals,
            'blocked_count': len(blocked),
            'market_summary': summary
        }

# Global trading engine
engine = TradingEngine()

# ============================================================
# API ROUTES
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Start price streaming on startup"""
    asyncio.create_task(price_streamer.start(engine.coins))

@app.on_event("shutdown")
async def shutdown_event():
    """Stop price streaming on shutdown"""
    price_streamer.stop()

@app.get("/")
async def root():
    return {
        "name": "Crypto AI Trading API",
        "version": "5.0.0",
        "features": [
            "Trade-type-specific analysis",
            "Experience level modifiers",
            "Enhanced FOMO detection",
            "Real-time WebSocket prices",
            "Expectancy & Readiness metrics"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "websocket_connected": price_streamer.running,
        "models_loaded": list(engine.models.keys()),
        "prices_available": list(price_streamer.prices.keys())
    }

@app.get("/prices")
async def get_prices():
    return price_streamer.prices

@app.get("/scan")
async def scan():
    return engine.scan_all()

@app.get("/analyze/{coin}")
async def analyze(
    coin: str,
    capital: float = 1000,
    trade_type: str = "SWING",
    experience: str = "INTERMEDIATE",
    reason: Optional[str] = None,
    recent_losses: int = 0,
    trades_today: int = 0,
    entry_price: Optional[float] = None
):
    try:
        # Validate trade_type
        trade_type_upper = trade_type.upper()
        if trade_type_upper not in ['SCALP', 'SHORT_TERM', 'SWING', 'INVESTMENT']:
            trade_type_upper = 'SWING'
        
        # Validate experience
        experience_upper = experience.upper()
        if experience_upper not in ['BEGINNER', 'INTERMEDIATE', 'ADVANCED']:
            experience_upper = 'INTERMEDIATE'
        
        # Validate reason
        reason_enum = None
        if reason:
            reason_upper = reason.upper()
            if reason_upper in ['STRATEGY', 'FOMO', 'NEWS', 'TIP', 'DIP_BUY']:
                reason_enum = TradeReason(reason_upper)
        
        request = AnalyzeRequest(
            coin=coin,
            capital=capital,
            trade_type=TradeType(trade_type_upper),
            experience=ExperienceLevel(experience_upper),
            reason=reason_enum,
            recent_losses=recent_losses,
            trades_today=trades_today,
            entry_price=entry_price
        )
        return engine.analyze(request)
    except Exception as e:
        print(f"❌ Analyze error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config/trade-types")
async def get_trade_types():
    """Get trade type configurations"""
    return TRADE_TYPE_CONFIG

@app.get("/config/experience")
async def get_experience_levels():
    """Get experience level configurations"""
    return EXPERIENCE_MODIFIERS

@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """WebSocket endpoint for live prices"""
    await websocket.accept()
    price_streamer.clients.add(websocket)
    
    try:
        # Send initial prices
        await websocket.send_json({
            'type': 'initial',
            'prices': price_streamer.prices
        })
        
        # Keep connection alive
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        price_streamer.clients.discard(websocket)

# ============================================================
# NEWS SERVICE
# ============================================================

class NewsService:
    """Service for fetching crypto and geopolitical news"""

    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.last_fetch = {}

        # Keywords for sentiment analysis
        self.bullish_keywords = [
            'surge', 'rally', 'bull', 'gain', 'rise', 'up', 'high', 'record',
            'adoption', 'approve', 'approved', 'etf', 'institutional', 'buy',
            'bullish', 'breakout', 'moon', 'pump', 'growth', 'positive'
        ]
        self.bearish_keywords = [
            'crash', 'drop', 'fall', 'bear', 'down', 'low', 'sell', 'dump',
            'ban', 'fraud', 'hack', 'scam', 'regulation', 'sec', 'lawsuit',
            'bearish', 'breakdown', 'fear', 'panic', 'negative', 'warning'
        ]

        # Geopolitical keywords that impact crypto
        self.geo_impact_keywords = [
            'federal reserve', 'fed', 'interest rate', 'inflation', 'war',
            'sanctions', 'china', 'russia', 'dollar', 'economy', 'recession',
            'bank', 'crisis', 'trade war', 'tariff', 'regulation'
        ]

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of news text"""
        text_lower = text.lower()

        bullish_count = sum(1 for word in self.bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in self.bearish_keywords if word in text_lower)

        if bullish_count > bearish_count:
            sentiment = 'BULLISH'
            score = min(bullish_count / 3, 1.0)
        elif bearish_count > bullish_count:
            sentiment = 'BEARISH'
            score = -min(bearish_count / 3, 1.0)
        else:
            sentiment = 'NEUTRAL'
            score = 0.0

        return {
            'sentiment': sentiment,
            'score': round(score, 2),
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count
        }

    def has_geo_impact(self, text: str) -> bool:
        """Check if news has geopolitical market impact"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.geo_impact_keywords)

    async def fetch_crypto_news(self) -> List[Dict]:
        """Fetch crypto news from multiple sources"""
        import aiohttp
        import xml.etree.ElementTree as ET

        cache_key = 'crypto_news'
        now = datetime.now().timestamp()

        # Check cache
        if cache_key in self.cache and cache_key in self.last_fetch:
            if now - self.last_fetch[cache_key] < self.cache_duration:
                return self.cache[cache_key]

        news_items = []

        # RSS feeds for crypto news
        rss_feeds = [
            ('https://cointelegraph.com/rss', 'CoinTelegraph'),
            ('https://coindesk.com/arc/outboundfeeds/rss/', 'CoinDesk'),
            ('https://decrypt.co/feed', 'Decrypt'),
            ('https://bitcoinmagazine.com/feed', 'Bitcoin Magazine'),
        ]

        async with aiohttp.ClientSession() as session:
            for feed_url, source in rss_feeds:
                try:
                    async with session.get(feed_url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            root = ET.fromstring(content)

                            # Parse RSS items
                            for item in root.findall('.//item')[:5]:  # Get top 5 from each
                                title = item.find('title')
                                link = item.find('link')
                                pub_date = item.find('pubDate')
                                description = item.find('description')

                                if title is not None:
                                    title_text = title.text or ''
                                    desc_text = (description.text or '')[:200] if description is not None else ''

                                    # Analyze sentiment
                                    sentiment = self.analyze_sentiment(title_text + ' ' + desc_text)

                                    news_items.append({
                                        'title': title_text,
                                        'source': source,
                                        'url': link.text if link is not None else '',
                                        'published': pub_date.text if pub_date is not None else '',
                                        'description': desc_text,
                                        'sentiment': sentiment['sentiment'],
                                        'sentiment_score': sentiment['score'],
                                        'category': 'CRYPTO',
                                        'has_market_impact': abs(sentiment['score']) > 0.3
                                    })
                except Exception as e:
                    print(f"Error fetching from {source}: {e}")

        # Sort by sentiment impact (most impactful first)
        news_items.sort(key=lambda x: abs(x['sentiment_score']), reverse=True)

        # Cache results
        self.cache[cache_key] = news_items[:20]  # Keep top 20
        self.last_fetch[cache_key] = now

        return self.cache[cache_key]

    async def fetch_geopolitical_news(self) -> List[Dict]:
        """Fetch geopolitical news that may impact crypto markets"""
        import aiohttp
        import xml.etree.ElementTree as ET

        cache_key = 'geo_news'
        now = datetime.now().timestamp()

        # Check cache
        if cache_key in self.cache and cache_key in self.last_fetch:
            if now - self.last_fetch[cache_key] < self.cache_duration:
                return self.cache[cache_key]

        news_items = []

        # RSS feeds for financial/geopolitical news
        rss_feeds = [
            ('https://feeds.bbci.co.uk/news/business/rss.xml', 'BBC Business'),
            ('https://rss.nytimes.com/services/xml/rss/nyt/Business.xml', 'NY Times Business'),
            ('https://feeds.reuters.com/reuters/businessNews', 'Reuters'),
            ('https://www.cnbc.com/id/100003114/device/rss/rss.html', 'CNBC'),
        ]

        async with aiohttp.ClientSession() as session:
            for feed_url, source in rss_feeds:
                try:
                    async with session.get(feed_url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            root = ET.fromstring(content)

                            # Parse RSS items
                            for item in root.findall('.//item')[:10]:
                                title = item.find('title')
                                link = item.find('link')
                                pub_date = item.find('pubDate')
                                description = item.find('description')

                                if title is not None:
                                    title_text = title.text or ''
                                    desc_text = (description.text or '')[:200] if description is not None else ''
                                    full_text = title_text + ' ' + desc_text

                                    # Only include if it has potential market impact
                                    if self.has_geo_impact(full_text):
                                        sentiment = self.analyze_sentiment(full_text)

                                        # Determine impact type
                                        impact_type = 'NEUTRAL'
                                        if any(w in full_text.lower() for w in ['war', 'crisis', 'sanctions', 'ban']):
                                            impact_type = 'HIGH_RISK'
                                        elif any(w in full_text.lower() for w in ['fed', 'interest rate', 'inflation']):
                                            impact_type = 'MONETARY'
                                        elif any(w in full_text.lower() for w in ['regulation', 'sec', 'law']):
                                            impact_type = 'REGULATORY'

                                        news_items.append({
                                            'title': title_text,
                                            'source': source,
                                            'url': link.text if link is not None else '',
                                            'published': pub_date.text if pub_date is not None else '',
                                            'description': desc_text,
                                            'sentiment': sentiment['sentiment'],
                                            'sentiment_score': sentiment['score'],
                                            'category': 'GEOPOLITICAL',
                                            'impact_type': impact_type,
                                            'has_market_impact': True
                                        })
                except Exception as e:
                    print(f"Error fetching from {source}: {e}")

        # Sort by recency (assuming pub_date parsing)
        self.cache[cache_key] = news_items[:15]  # Keep top 15
        self.last_fetch[cache_key] = now

        return self.cache[cache_key]

    async def get_market_sentiment_summary(self) -> Dict:
        """Get overall market sentiment from news"""
        crypto_news = await self.fetch_crypto_news()
        geo_news = await self.fetch_geopolitical_news()

        all_news = crypto_news + geo_news

        if not all_news:
            return {
                'overall_sentiment': 'NEUTRAL',
                'sentiment_score': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'high_impact_count': 0
            }

        bullish = sum(1 for n in all_news if n['sentiment'] == 'BULLISH')
        bearish = sum(1 for n in all_news if n['sentiment'] == 'BEARISH')
        neutral = sum(1 for n in all_news if n['sentiment'] == 'NEUTRAL')
        high_impact = sum(1 for n in all_news if n.get('has_market_impact', False))

        avg_score = sum(n['sentiment_score'] for n in all_news) / len(all_news)

        if avg_score > 0.2:
            overall = 'BULLISH'
        elif avg_score < -0.2:
            overall = 'BEARISH'
        else:
            overall = 'NEUTRAL'

        return {
            'overall_sentiment': overall,
            'sentiment_score': round(avg_score, 2),
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
            'high_impact_count': high_impact,
            'total_news': len(all_news)
        }

# Global news service
news_service = NewsService()

# ============================================================
# NEWS API ROUTES
# ============================================================

@app.get("/news/crypto")
async def get_crypto_news():
    """Get latest crypto news with sentiment analysis"""
    try:
        news = await news_service.fetch_crypto_news()
        sentiment = await news_service.get_market_sentiment_summary()
        return {
            'news': news,
            'market_sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error fetching crypto news: {e}")
        return {
            'news': [],
            'market_sentiment': {'overall_sentiment': 'UNKNOWN'},
            'error': str(e)
        }

@app.get("/news/geopolitical")
async def get_geopolitical_news():
    """Get geopolitical news that may impact crypto markets"""
    try:
        news = await news_service.fetch_geopolitical_news()
        return {
            'news': news,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error fetching geopolitical news: {e}")
        return {
            'news': [],
            'error': str(e)
        }

@app.get("/news/all")
async def get_all_news():
    """Get all news (crypto + geopolitical) with sentiment summary"""
    try:
        crypto_news = await news_service.fetch_crypto_news()
        geo_news = await news_service.fetch_geopolitical_news()
        sentiment = await news_service.get_market_sentiment_summary()

        return {
            'crypto_news': crypto_news,
            'geopolitical_news': geo_news,
            'market_sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error fetching news: {e}")
        return {
            'crypto_news': [],
            'geopolitical_news': [],
            'market_sentiment': {'overall_sentiment': 'UNKNOWN'},
            'error': str(e)
        }

@app.get("/news/sentiment")
async def get_news_sentiment():
    """Get market sentiment summary from news"""
    try:
        sentiment = await news_service.get_market_sentiment_summary()
        return sentiment
    except Exception as e:
        return {'overall_sentiment': 'UNKNOWN', 'error': str(e)}

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Crypto AI Trading API v5.0")
    print("📊 Trade-Type-Specific Analysis System")
    print("📰 News Feed Service Enabled")
    uvicorn.run(app, host="0.0.0.0", port=8000)