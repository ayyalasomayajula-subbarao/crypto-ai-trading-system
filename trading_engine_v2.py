import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LAYER 1: PRICE FORECAST ENGINE (Model A - Narrative)
# ============================================================

class PriceForecastEngine:
    """
    Predicts WHERE price is likely to go.
    Used for: Narrative, warnings, scenarios
    """
    
    def __init__(self, coin):
        self.coin = coin
        self.model_dir = f"models/{coin}"
        self.loaded = False
        self.load_models()
    
    def load_models(self):
        try:
            self.rf_model = joblib.load(f"{self.model_dir}/random_forest.pkl")
            self.xgb_model = joblib.load(f"{self.model_dir}/xgboost.pkl")
            self.label_encoder = joblib.load(f"{self.model_dir}/label_encoder.pkl")
            
            with open(f"{self.model_dir}/feature_list.txt", 'r') as f:
                self.feature_cols = [line.strip() for line in f.readlines()]
            
            self.classes = ['DOWN', 'SIDEWAYS', 'UP']
            self.loaded = True
        except Exception as e:
            print(f"Warning: Could not load forecast model for {self.coin}: {e}")
            self.loaded = False
    
    def forecast(self, features):
        if not self.loaded:
            return None
        
        X = features[self.feature_cols] if isinstance(features, pd.DataFrame) else features
        
        # Get probabilities
        rf_proba = self.rf_model.predict_proba(X)
        rf_classes = list(self.rf_model.classes_)
        
        rf_ordered = np.zeros((len(X), 3))
        for i, cls in enumerate(self.classes):
            if cls in rf_classes:
                rf_ordered[:, i] = rf_proba[:, rf_classes.index(cls)]
        
        xgb_proba = self.xgb_model.predict_proba(X)
        xgb_classes = list(self.label_encoder.classes_)
        
        xgb_ordered = np.zeros((len(X), 3))
        for i, cls in enumerate(self.classes):
            if cls in xgb_classes:
                xgb_ordered[:, i] = xgb_proba[:, xgb_classes.index(cls)]
        
        ensemble_proba = rf_ordered * 0.5 + xgb_ordered * 0.5
        p = ensemble_proba[-1] if len(ensemble_proba.shape) > 1 else ensemble_proba
        
        direction_idx = p.argmax()
        direction_bias = self.classes[direction_idx]
        
        # Volatility estimates per coin
        moves = {
            'BTC_USDT': {'up': 5, 'down': 4},
            'ETH_USDT': {'up': 7, 'down': 5},
            'SOL_USDT': {'up': 10, 'down': 8},
            'PEPE_USDT': {'up': 15, 'down': 12}
        }.get(self.coin, {'up': 5, 'down': 4})
        
        return {
            'direction_bias': direction_bias,
            'confidence': round(float(p[direction_idx]), 3),
            'probabilities': {
                'UP': round(float(p[2]), 3),
                'SIDEWAYS': round(float(p[1]), 3),
                'DOWN': round(float(p[0]), 3)
            },
            'scenarios': {
                'bull_case': {'move': f"+{moves['up']}%", 'probability': round(float(p[2]), 3)},
                'base_case': {'move': 'SIDEWAYS', 'probability': round(float(p[1]), 3)},
                'bear_case': {'move': f"-{moves['down']}%", 'probability': round(float(p[0]), 3)}
            }
        }


# ============================================================
# LAYER 2: DECISION ENGINE (Model B - WIN/LOSS/NO_TRADE)
# ============================================================

class DecisionEngine:
    """
    Decides WHETHER to trade based on WIN probability.
    Uses the new Decision Model trained on TP/SL outcomes.
    """
    
    def __init__(self, coin):
        self.coin = coin
        self.model_dir = f"models/{coin}"
        self.loaded = False
        self.load_model()
        
        # Confidence thresholds (based on your results!)
        self.thresholds = {
            'BTC_USDT': {'high': 0.45, 'medium': 0.40, 'low': 0.35},
            'ETH_USDT': {'high': 0.45, 'medium': 0.40, 'low': 0.35},
            'SOL_USDT': {'high': 0.45, 'medium': 0.40, 'low': 0.35},
            'PEPE_USDT': {'high': 0.50, 'medium': 0.45, 'low': 0.40}  # Higher for PEPE
        }
    
    def load_model(self):
        try:
            self.model = joblib.load(f"{self.model_dir}/decision_model.pkl")
            
            with open(f"{self.model_dir}/decision_features.txt", 'r') as f:
                self.feature_cols = [line.strip() for line in f.readlines()]
            
            self.classes = list(self.model.classes_)
            self.loaded = True
        except Exception as e:
            print(f"Warning: Could not load decision model for {self.coin}: {e}")
            self.loaded = False
    
    def get_win_probability(self, features):
        """Get probability of WIN (TP hit before SL)"""
        if not self.loaded:
            return None, None
        
        X = features[self.feature_cols] if isinstance(features, pd.DataFrame) else features
        
        probabilities = self.model.predict_proba(X)
        
        # Get WIN probability
        win_idx = self.classes.index('WIN') if 'WIN' in self.classes else 0
        loss_idx = self.classes.index('LOSS') if 'LOSS' in self.classes else 1
        no_trade_idx = self.classes.index('NO_TRADE') if 'NO_TRADE' in self.classes else 2
        
        p = probabilities[-1] if len(probabilities.shape) > 1 else probabilities
        
        return {
            'WIN': float(p[win_idx]),
            'LOSS': float(p[loss_idx]),
            'NO_TRADE': float(p[no_trade_idx])
        }, p
    
    def decide(self, features, user_input, forecast=None):
        """
        Make trading decision based on WIN probability.
        
        This is the KEY change - we now use WIN probability, not direction!
        """
        win_probs, raw_proba = self.get_win_probability(features)
        
        if win_probs is None:
            return self._no_signal("Decision model not loaded")
        
        win_prob = win_probs['WIN']
        loss_prob = win_probs['LOSS']
        no_trade_prob = win_probs['NO_TRADE']
        
        position_status = user_input.get('position_status', 'NO_POSITION')
        thresholds = self.thresholds.get(self.coin, {'high': 0.45, 'medium': 0.40, 'low': 0.35})
        
        reasoning = []
        warnings = []
        
        # ============================================
        # NO POSITION - Looking to BUY
        # ============================================
        if position_status == 'NO_POSITION':
            
            # HIGH confidence BUY
            if win_prob >= thresholds['high']:
                reasoning.append(f"High WIN probability: {win_prob*100:.1f}%")
                reasoning.append(f"Expected to hit TP before SL")
                
                if forecast and forecast['direction_bias'] == 'UP':
                    reasoning.append("Confirmed by bullish forecast")
                elif forecast and forecast['direction_bias'] == 'DOWN':
                    warnings.append("Forecast shows bearish bias - use caution")
                
                return {
                    'verdict': 'BUY',
                    'confidence': 'HIGH',
                    'win_probability': round(win_prob, 3),
                    'loss_probability': round(loss_prob, 3),
                    'reasoning': reasoning,
                    'warnings': warnings,
                    'expected_outcome': 'WIN'
                }
            
            # MEDIUM confidence - conditional BUY
            elif win_prob >= thresholds['medium']:
                reasoning.append(f"Moderate WIN probability: {win_prob*100:.1f}%")
                
                # Check forecast for confirmation
                if forecast and forecast['direction_bias'] == 'UP':
                    reasoning.append("Supported by bullish forecast")
                    return {
                        'verdict': 'BUY',
                        'confidence': 'MEDIUM',
                        'win_probability': round(win_prob, 3),
                        'loss_probability': round(loss_prob, 3),
                        'reasoning': reasoning,
                        'warnings': ["Use smaller position size"],
                        'expected_outcome': 'WIN'
                    }
                else:
                    reasoning.append("No bullish confirmation from forecast")
                    return {
                        'verdict': 'WAIT',
                        'confidence': 'MEDIUM',
                        'win_probability': round(win_prob, 3),
                        'loss_probability': round(loss_prob, 3),
                        'reasoning': reasoning,
                        'warnings': ["Wait for better entry or confirmation"],
                        'suggested_action': 'Monitor for higher WIN probability'
                    }
            
            # LOW confidence - check if LOSS is high
            elif loss_prob >= 0.50:
                reasoning.append(f"High LOSS probability: {loss_prob*100:.1f}%")
                reasoning.append("SL likely to be hit before TP")
                
                return {
                    'verdict': 'WAIT',
                    'confidence': 'HIGH',
                    'win_probability': round(win_prob, 3),
                    'loss_probability': round(loss_prob, 3),
                    'reasoning': reasoning,
                    'warnings': ["⚠️ High risk of loss - do not enter"],
                    'expected_outcome': 'LOSS'
                }
            
            # Uncertain
            else:
                reasoning.append(f"WIN: {win_prob*100:.1f}%, LOSS: {loss_prob*100:.1f}%")
                reasoning.append("No clear edge")
                
                return {
                    'verdict': 'WAIT',
                    'confidence': 'LOW',
                    'win_probability': round(win_prob, 3),
                    'loss_probability': round(loss_prob, 3),
                    'reasoning': reasoning,
                    'warnings': [],
                    'suggested_action': 'Wait for clearer signal'
                }
        
        # ============================================
        # IN POSITION - Looking to HOLD or EXIT
        # ============================================
        else:
            entry_price = user_input.get('entry_price', 0)
            current_price = user_input.get('current_price', 0)
            
            if entry_price > 0 and current_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price * 100
                reasoning.append(f"Current P&L: {pnl_pct:+.2f}%")
            else:
                pnl_pct = 0
            
            # High LOSS probability - EXIT
            if loss_prob >= 0.50:
                reasoning.append(f"High LOSS probability: {loss_prob*100:.1f}%")
                
                return {
                    'verdict': 'EXIT',
                    'confidence': 'HIGH',
                    'win_probability': round(win_prob, 3),
                    'loss_probability': round(loss_prob, 3),
                    'reasoning': reasoning,
                    'warnings': ["Risk of further downside"],
                    'suggested_action': 'Close position to protect capital'
                }
            
            # High WIN probability - HOLD
            elif win_prob >= thresholds['medium']:
                reasoning.append(f"WIN probability still good: {win_prob*100:.1f}%")
                
                return {
                    'verdict': 'HOLD',
                    'confidence': 'HIGH' if win_prob >= thresholds['high'] else 'MEDIUM',
                    'win_probability': round(win_prob, 3),
                    'loss_probability': round(loss_prob, 3),
                    'reasoning': reasoning,
                    'warnings': [],
                    'suggested_action': 'Continue holding, trail stop loss'
                }
            
            # Uncertain - depends on P&L
            else:
                if pnl_pct > 3:
                    return {
                        'verdict': 'HOLD',
                        'confidence': 'LOW',
                        'win_probability': round(win_prob, 3),
                        'loss_probability': round(loss_prob, 3),
                        'reasoning': reasoning + ["In profit, no clear exit signal"],
                        'warnings': ["Consider taking partial profit"],
                        'suggested_action': 'Trail stop to breakeven'
                    }
                elif pnl_pct < -2:
                    return {
                        'verdict': 'EXIT',
                        'confidence': 'MEDIUM',
                        'win_probability': round(win_prob, 3),
                        'loss_probability': round(loss_prob, 3),
                        'reasoning': reasoning + ["In loss, weak signal"],
                        'warnings': ["Cut losses"],
                        'suggested_action': 'Exit to preserve capital'
                    }
                else:
                    return {
                        'verdict': 'HOLD',
                        'confidence': 'LOW',
                        'win_probability': round(win_prob, 3),
                        'loss_probability': round(loss_prob, 3),
                        'reasoning': reasoning + ["Near breakeven, no clear signal"],
                        'warnings': [],
                        'suggested_action': 'Monitor closely'
                    }
    
    def _no_signal(self, reason):
        return {
            'verdict': 'WAIT',
            'confidence': 'LOW',
            'win_probability': 0,
            'loss_probability': 0,
            'reasoning': [reason],
            'warnings': ["Model not available"]
        }


# ============================================================
# LAYER 3: RISK ENGINE
# ============================================================

class RiskEngine:
    """Position sizing based on WIN probability"""
    
    def calculate_risk(self, decision, user_input):
        verdict = decision.get('verdict', 'WAIT')
        win_prob = decision.get('win_probability', 0)
        confidence = decision.get('confidence', 'LOW')
        
        capital = user_input.get('capital', 1000)
        risk_pref = user_input.get('risk_preference', 'MEDIUM')
        coin = user_input.get('coin', 'BTC_USDT')
        
        if verdict in ['WAIT', 'EXIT']:
            return {
                'position_size_pct': 0,
                'position_size_usd': 0,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.05,
                'risk_reward_ratio': 1.67,
                'max_loss_usd': 0,
                'risk_level': 'NONE',
                'action': 'NO_TRADE' if verdict == 'WAIT' else 'CLOSE_POSITION'
            }
        
        # Position sizing based on WIN probability (Kelly-inspired)
        # Higher WIN prob = larger position
        if win_prob >= 0.50:
            base_size = 0.50  # 50%
        elif win_prob >= 0.45:
            base_size = 0.40  # 40%
        elif win_prob >= 0.40:
            base_size = 0.30  # 30%
        else:
            base_size = 0.20  # 20%
        
        # Adjust by risk preference
        risk_mult = {'LOW': 0.5, 'MEDIUM': 1.0, 'HIGH': 1.3}.get(risk_pref, 1.0)
        
        position_size_pct = min(base_size * risk_mult, 0.50)
        position_size_usd = capital * position_size_pct
        
        # TP/SL based on coin volatility
        volatility = {
            'BTC_USDT': {'sl': 0.03, 'tp': 0.05},
            'ETH_USDT': {'sl': 0.035, 'tp': 0.055},
            'SOL_USDT': {'sl': 0.04, 'tp': 0.06},
            'PEPE_USDT': {'sl': 0.05, 'tp': 0.08}
        }.get(coin, {'sl': 0.03, 'tp': 0.05})
        
        stop_loss_pct = volatility['sl']
        take_profit_pct = volatility['tp']
        
        max_loss_usd = position_size_usd * stop_loss_pct
        rr_ratio = take_profit_pct / stop_loss_pct
        
        risk_level = 'HIGH' if position_size_pct > 0.40 else ('MEDIUM' if position_size_pct > 0.25 else 'LOW')
        
        return {
            'position_size_pct': round(position_size_pct, 3),
            'position_size_usd': round(position_size_usd, 2),
            'stop_loss_pct': round(stop_loss_pct, 4),
            'take_profit_pct': round(take_profit_pct, 4),
            'risk_reward_ratio': round(rr_ratio, 2),
            'max_loss_usd': round(max_loss_usd, 2),
            'risk_level': risk_level,
            'action': 'OPEN_POSITION'
        }


# ============================================================
# MAIN TRADING ENGINE V2
# ============================================================

class TradingEngineV2:
    """
    Trading Engine V2 - Uses Decision Model for WIN/LOSS predictions
    """
    
    def __init__(self):
        self.forecast_engines = {}
        self.decision_engines = {}
        self.risk_engine = RiskEngine()
        
        for coin in ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']:
            self.forecast_engines[coin] = PriceForecastEngine(coin)
            self.decision_engines[coin] = DecisionEngine(coin)
    
    def analyze(self, coin, features, user_input):
        """Complete analysis using both Forecast and Decision models"""
        user_input['coin'] = coin
        
        # Layer 1: Forecast (for narrative)
        forecast_engine = self.forecast_engines.get(coin)
        forecast = forecast_engine.forecast(features) if forecast_engine and forecast_engine.loaded else None
        
        # Layer 2: Decision (for BUY/WAIT/EXIT)
        decision_engine = self.decision_engines.get(coin)
        if not decision_engine or not decision_engine.loaded:
            return {'error': f'Decision model not loaded for {coin}'}
        
        decision = decision_engine.decide(features, user_input, forecast)
        
        # Layer 3: Risk
        risk = self.risk_engine.calculate_risk(decision, user_input)
        
        return {
            'coin': coin,
            'timestamp': datetime.now().isoformat(),
            
            # Main verdict
            'verdict': decision['verdict'],
            'confidence': decision['confidence'],
            
            # Probabilities (KEY CHANGE!)
            'win_probability': decision['win_probability'],
            'loss_probability': decision['loss_probability'],
            
            # Forecast (for context)
            'forecast': forecast,
            
            # Reasoning
            'reasoning': decision.get('reasoning', []),
            'suggested_action': decision.get('suggested_action', ''),
            'expected_outcome': decision.get('expected_outcome', 'UNCERTAIN'),
            
            # Risk management
            'risk': risk,
            
            # Warnings
            'warnings': decision.get('warnings', [])
        }


# ============================================================
# TEST
# ============================================================

def test_trading_engine_v2():
    print("\n" + "#"*70)
    print("  TRADING ENGINE V2 TEST (Decision Model)")
    print("#"*70)
    
    engine = TradingEngineV2()
    coins = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']
    
    for coin in coins:
        print(f"\n{'='*70}")
        print(f"  {coin}")
        print(f"{'='*70}")
        
        df = pd.read_csv(f"data/{coin}_multi_tf_features.csv")
        latest = df.tail(1)
        
        # Test: Want to BUY
        print(f"\n  📊 SCENARIO: Want to BUY")
        print(f"  {'-'*50}")
        
        user_input = {
            'position_status': 'NO_POSITION',
            'capital': 1000,
            'risk_preference': 'MEDIUM'
        }
        
        result = engine.analyze(coin, latest, user_input)
        
        print(f"  Verdict: {result['verdict']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  WIN Probability: {result['win_probability']*100:.1f}%")
        print(f"  LOSS Probability: {result['loss_probability']*100:.1f}%")
        print(f"  Reasoning: {result['reasoning'][:2]}")
        
        if result['verdict'] == 'BUY':
            print(f"\n  💰 Risk Management:")
            print(f"     Position: ${result['risk']['position_size_usd']:.0f} ({result['risk']['position_size_pct']*100:.0f}%)")
            print(f"     Stop Loss: {result['risk']['stop_loss_pct']*100:.1f}%")
            print(f"     Take Profit: {result['risk']['take_profit_pct']*100:.1f}%")
        
        if result['warnings']:
            print(f"\n  ⚠️ Warnings: {result['warnings']}")
        
        # Forecast context
        if result['forecast']:
            print(f"\n  📈 Forecast: {result['forecast']['direction_bias']} ({result['forecast']['confidence']*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("🎉 TRADING ENGINE V2 TEST COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_trading_engine_v2()