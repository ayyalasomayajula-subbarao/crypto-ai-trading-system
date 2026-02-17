"""
Crypto AI Trading API - v6.0
Complete Trade-Type-Specific Analysis System with AI-Powered Insights

FEATURES in v6.0:
- Everything from v5.0
- Groq as primary AI provider (Llama 3.3 70B)
- OpenAI as fallback AI provider (GPT-3.5-turbo)
- AI-powered trade analysis with natural language insights
- Structured JSON responses from LLM
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
import requests
from dotenv import load_dotenv
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, ForceIndexIndicator

# Load environment variables
load_dotenv()

# ============================================================
# AI SERVICE (Groq FREE + OpenAI Fallback)
# ============================================================

class AIService:
    """AI-powered analysis using Groq (free) as primary and OpenAI as fallback"""

    def __init__(self):
        self.groq_client = None
        self.openai_client = None
        self.groq_available = False
        self.openai_available = False

        # Initialize Groq
        groq_key = os.getenv('GROQ_API_KEY', '')
        if groq_key and groq_key != 'gsk_your_groq_key_here':
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
                self.groq_available = True
                print("  Groq (FREE):    Ready")
            except ImportError:
                print("  Groq (FREE):    Not installed (pip install groq)")
            except Exception as e:
                print(f"  Groq (FREE):    Error - {e}")
        else:
            print("  Groq (FREE):    No API key set")

        # Initialize OpenAI
        openai_key = os.getenv('OPENAI_API_KEY', '')
        if openai_key and openai_key != 'your_openai_key_here':
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)
                self.openai_available = True
                print("  OpenAI:         Ready")
            except ImportError:
                print("  OpenAI:         Not installed (pip install openai)")
            except Exception as e:
                print(f"  OpenAI:         Error - {e}")
        else:
            print("  OpenAI:         No API key set")

    def build_prompt(self, coin: str, analysis_data: Dict, market_sentiment: Dict = None, news_data: Dict = None, derivatives_data: Dict = None, whale_data: Dict = None) -> str:
        """Build a comprehensive prompt for deep AI analysis"""
        price = analysis_data.get('price', 0)
        verdict = analysis_data.get('verdict', 'UNKNOWN')
        win_prob = analysis_data.get('win_probability', 0)
        loss_prob = analysis_data.get('loss_probability', 0)
        sideways_prob = analysis_data.get('sideways_probability', 0)
        expectancy = analysis_data.get('expectancy', 0)
        readiness = analysis_data.get('readiness', 0)
        trade_type = analysis_data.get('trade_type', 'SWING')
        experience = analysis_data.get('experience_level', 'INTERMEDIATE')
        capital = analysis_data.get('capital', 1000)

        regime = analysis_data.get('market_context', {}).get('regime', {})
        adx = regime.get('adx', 0)
        market_regime = regime.get('regime', 'UNKNOWN')
        volatility = regime.get('volatility', 'UNKNOWN')
        volatility_pct = regime.get('volatility_pct', 0)

        btc = analysis_data.get('market_context', {}).get('btc', {})
        btc_trend = btc.get('overall_trend', 'NEUTRAL')
        btc_price = btc.get('price', 0)
        btc_change = btc.get('change_24h', 0)

        reasoning = analysis_data.get('reasoning', [])
        warnings = analysis_data.get('warnings', [])

        forecast = analysis_data.get('forecast', {})
        bull_target = forecast.get('bull_target', 0)
        bear_target = forecast.get('bear_target', 0)

        # Risk management data
        risk = analysis_data.get('risk', {})
        risk_section = ""
        if risk.get('action') == 'OPEN_POSITION':
            risk_section = f"""
POSITION SIZING:
- Suggested Position: ${risk.get('position_size_usd', 0):,.2f} ({risk.get('position_size_pct', 0)}% of capital)
- Entry Price: ${risk.get('entry_price', 0):,.8g}
- Stop Loss: ${risk.get('stop_loss_price', 0):,.8g} (-{risk.get('stop_loss_pct', 0)}%)
- Take Profit: ${risk.get('take_profit_price', 0):,.8g} (+{risk.get('take_profit_pct', 0)}%)
- Max Loss: ${risk.get('max_loss_usd', 0):,.2f}
- Risk/Reward: {risk.get('risk_reward', 'N/A')}"""

        # Volume analysis data
        vol = analysis_data.get('volume_analysis', {})
        vol_signal = vol.get('overall_signal', 'N/A')
        vol_summary = vol.get('summary', 'No volume data')
        obv = vol.get('obv', {})
        mfi = vol.get('mfi', {})
        delta = vol.get('buy_sell_delta', {})
        spikes = vol.get('volume_spikes', {})

        # Volume Profile data
        vp = vol.get('volume_profile', {})
        poc = vp.get('poc', {})
        hvn_list = vp.get('hvn', [])
        lvn_list = vp.get('lvn', [])

        vp_section = ""
        if poc.get('price', 0) > 0:
            hvn_prices = ', '.join([formatPrice(h['price']) for h in hvn_list[:5]]) if hvn_list else 'None'
            lvn_prices = ', '.join([formatPrice(l['price']) for l in lvn_list[:5]]) if lvn_list else 'None'
            vp_section = f"""
- Volume Profile (7d):
  * POC (Fair Value): {formatPrice(poc.get('price', 0))} ({poc.get('volume_pct', 0):.1f}% of volume)
  * Value Area: {formatPrice(vp.get('value_area_low', 0))} - {formatPrice(vp.get('value_area_high', 0))}
  * Price vs POC: {vp.get('price_vs_poc', 'N/A')}
  * HVN (Support/Resistance): {hvn_prices}
  * LVN (Fast-move zones): {lvn_prices}
  * Analysis: {vp.get('analysis', 'N/A')}"""

        volume_section = f"""
VOLUME ANALYSIS:
- Overall Volume Signal: {vol_signal}
- Summary: {vol_summary}
- OBV Trend: {obv.get('trend', 'N/A')} | Divergence: {obv.get('divergence', 'NONE')}
- MFI: {mfi.get('value', 0):.1f} ({mfi.get('zone', 'N/A')}) - {mfi.get('interpretation', 'N/A')}
- Buy/Sell Delta: {delta.get('delta_pct', 0):.1f}% ({delta.get('pressure', 'N/A')}) | Strength: {delta.get('strength', 'N/A')}
- Volume Spike Ratio: {spikes.get('current_ratio', 0):.2f}x | Active Spike: {'YES' if spikes.get('is_spike') else 'NO'}
- Spike Count (24h): {spikes.get('spike_count_24h', 0)}{vp_section}"""

        # Market sentiment section
        sentiment_section = ""
        if market_sentiment:
            fg = market_sentiment.get('fear_greed', {})
            fr = market_sentiment.get('funding_rate', {})
            oi = market_sentiment.get('open_interest', {})
            overall_sent = market_sentiment.get('overall_sentiment', 'UNKNOWN')
            sentiment_section = f"""
MARKET SENTIMENT DATA:
- Overall Market Sentiment: {overall_sent}
- Fear & Greed Index: {fg.get('value', 'N/A')} ({fg.get('label', 'N/A')}) | Trend: {fg.get('trend', 'N/A')}
- Funding Rate: {fr.get('rate_pct', 'N/A')} ({fr.get('sentiment', 'N/A')}) - {fr.get('interpretation', 'N/A')}
- Open Interest: {oi.get('formatted', 'N/A')}"""

        # News section
        news_section = ""
        if news_data:
            news_sentiment = news_data.get('market_sentiment', {})
            news_items = news_data.get('crypto_news', [])[:8]
            geo_items = news_data.get('geopolitical_news', [])[:4]
            news_section = f"""
NEWS & HEADLINES:
- News Sentiment: {news_sentiment.get('overall_sentiment', 'NEUTRAL')} (Score: {news_sentiment.get('sentiment_score', 0)})
- Bullish Headlines: {news_sentiment.get('bullish_count', 0)} | Bearish: {news_sentiment.get('bearish_count', 0)} | High Impact: {news_sentiment.get('high_impact_count', 0)}"""
            if news_items:
                news_section += "\n- Recent Crypto Headlines:"
                for n in news_items[:6]:
                    news_section += f"\n  * [{n.get('sentiment', 'NEUTRAL')}] {n.get('title', '')}"
            if geo_items:
                news_section += "\n- Geopolitical Headlines:"
                for n in geo_items[:3]:
                    news_section += f"\n  * [{n.get('impact_type', 'NEUTRAL')}] {n.get('title', '')}"

        # Derivatives data section
        derivatives_section = ""
        if derivatives_data and derivatives_data.get('overall_signal') != 'ERROR':
            ls = derivatives_data.get('long_short_ratio', {})
            tk = derivatives_data.get('taker_volume', {})
            ob = derivatives_data.get('order_book', {})
            liq = derivatives_data.get('liquidations', {})
            top = ls.get('top_traders', {})
            glb = ls.get('global', {})
            derivatives_section = f"""
DERIVATIVES INTELLIGENCE:
- Overall Derivatives Signal: {derivatives_data.get('overall_signal', 'N/A')}
- L/S Ratio (Top Traders): {top.get('long_pct', 50):.1f}% Long / {top.get('short_pct', 50):.1f}% Short | Trend: {top.get('trend', 'N/A')}
- L/S Ratio (Global): {glb.get('long_pct', 50):.1f}% Long / {glb.get('short_pct', 50):.1f}% Short | Trend: {glb.get('trend', 'N/A')}
- Smart Money Signal: {ls.get('signal', 'N/A')}
- Taker Buy/Sell: Ratio {tk.get('ratio', 1.0):.3f} | Pressure: {tk.get('pressure', 'N/A')} | Trend: {tk.get('trend', 'N/A')}
- Order Book: Bid/Ask Ratio {ob.get('bid_ask_ratio', 1.0):.3f} | Imbalance: {ob.get('imbalance', 'N/A')}
- Bid Wall: {formatPrice(ob.get('strongest_bid', {}).get('price', 0))} (${ob.get('strongest_bid', {}).get('size_usd', 0):,.0f})
- Ask Wall: {formatPrice(ob.get('strongest_ask', {}).get('price', 0))} (${ob.get('strongest_ask', {}).get('size_usd', 0):,.0f})
- OI Change (24h): {liq.get('oi_change_pct', 0):.1f}% | Signal: {liq.get('recent_signal', 'NONE')}
- Nearest Long Liq (100x): {formatPrice(liq.get('nearest_long_liq', {}).get('price', 0)) if liq.get('nearest_long_liq') else 'N/A'}
- Nearest Short Liq (100x): {formatPrice(liq.get('nearest_short_liq', {}).get('price', 0)) if liq.get('nearest_short_liq') else 'N/A'}"""

        # Whale activity section
        whale_section = ""
        if whale_data and whale_data.get('whale_signal') != 'ERROR':
            whale_section = f"""
WHALE ACTIVITY:
- Whale Signal: {whale_data.get('whale_signal', 'NEUTRAL')}
- Large Trade Count: {whale_data.get('large_trade_count', 0)}
- Alert: {whale_data.get('alert', 'None')}"""
            txs = whale_data.get('recent_large_txs', [])
            if txs:
                whale_section += "\n- Recent Large Transactions:"
                for tx in txs[:5]:
                    whale_section += f"\n  * {tx.get('value_display', 'N/A')} at {tx.get('time', 'N/A')}"

        coin_display = coin.replace('_', '/')
        prompt = f"""You are a top-tier crypto trading analyst providing institutional-grade analysis. Analyze ALL the data below and provide a comprehensive, decision-ready analysis for {coin_display}.

Think deeply about the interplay between technical indicators, volume flow, market sentiment, macro conditions, and news. Consider what experienced traders would look for.

=== CORE DATA ===

COIN: {coin_display}
CURRENT PRICE: ${price:,.8g}
TRADE TYPE: {trade_type} | EXPERIENCE: {experience} | CAPITAL: ${capital:,.2f}

MODEL ANALYSIS:
- System Verdict: {verdict}
- WIN Probability: {win_prob:.1f}% | LOSS: {loss_prob:.1f}% | SIDEWAYS: {sideways_prob:.1f}%
- Expectancy (WIN-LOSS): {expectancy:.1f}%
- Readiness (WIN-Threshold): {readiness:.1f}%

MARKET CONDITIONS:
- ADX (Trend Strength): {adx:.1f} | Market Regime: {market_regime}
- Volatility: {volatility} ({volatility_pct:.1f}%)
- BTC: ${btc_price:,.2f} ({btc_change:+.1f}% 24h) | Trend: {btc_trend}
{risk_section}
{volume_section}
{sentiment_section}
{news_section}
{derivatives_section}
{whale_section}

PRICE TARGETS: Bull ${bull_target:,.8g} | Bear ${bear_target:,.8g}
SYSTEM REASONING: {'; '.join(reasoning)}
WARNINGS: {'; '.join(warnings) if warnings else 'None'}

=== INSTRUCTIONS ===

Provide a DEEP analysis. Do NOT just restate the numbers — interpret them, find patterns, explain what they mean together, and give the trader a clear decision framework.

Respond with ONLY valid JSON (no markdown, no code blocks) in this exact format:
{{
  "ai_verdict": "BUY" or "WAIT" or "AVOID",
  "confidence_score": 1-10,
  "tldr": "2-3 sentence executive summary that captures the key situation and recommended action. Be specific about price levels and probabilities.",
  "key_insights": [
    "Insight 1: A specific observation combining multiple data points (e.g. volume + price action + sentiment)",
    "Insight 2: Another cross-referenced insight",
    "Insight 3: A third actionable insight"
  ],
  "positives": [
    {{
      "title": "Short title (e.g. 'Strong Volume Accumulation')",
      "detail": "Detailed explanation of why this is bullish, referencing specific data points. 2-3 sentences."
    }},
    {{
      "title": "Another positive factor",
      "detail": "Detailed explanation. 2-3 sentences."
    }},
    {{
      "title": "Third positive",
      "detail": "Detailed explanation. 2-3 sentences."
    }}
  ],
  "risks": [
    {{
      "title": "Short title (e.g. 'Bearish Momentum Persists')",
      "detail": "Detailed explanation of the risk, referencing specific data. 2-3 sentences.",
      "severity": "HIGH" or "MEDIUM" or "LOW"
    }},
    {{
      "title": "Another risk factor",
      "detail": "Detailed explanation. 2-3 sentences.",
      "severity": "HIGH" or "MEDIUM" or "LOW"
    }},
    {{
      "title": "Third risk",
      "detail": "Detailed explanation. 2-3 sentences.",
      "severity": "HIGH" or "MEDIUM" or "LOW"
    }}
  ],
  "market_pulse": {{
    "trend_summary": "1 sentence on the current trend direction and strength",
    "volume_verdict": "1 sentence on what volume is telling us",
    "sentiment_verdict": "1 sentence on overall market mood from sentiment data and news",
    "macro_context": "1 sentence on BTC influence and broader market conditions"
  }},
  "entry_strategy": "Specific, actionable entry strategy with price levels. Include when to enter, where to set stops, and profit targets.",
  "what_to_watch": ["Specific trigger or level 1", "Specific trigger or level 2", "Specific trigger or level 3"],
  "sentiment_read": "BULLISH" or "BEARISH" or "NEUTRAL",
  "conviction_reason": "The single most important factor driving your verdict. Be specific."
}}"""
        return prompt

    async def get_ai_analysis(self, coin: str, analysis_data: Dict, market_sentiment: Dict = None, news_data: Dict = None, derivatives_data: Dict = None, whale_data: Dict = None) -> Dict:
        """Get AI analysis using Groq (primary) or OpenAI (fallback)"""
        prompt = self.build_prompt(coin, analysis_data, market_sentiment, news_data, derivatives_data, whale_data)

        # Try Groq first (FREE)
        if self.groq_available:
            try:
                result = await self._call_groq(prompt)
                if result:
                    result['ai_provider'] = 'Groq (Llama 3.3 70B)'
                    result['ai_cost'] = 'FREE'
                    return result
            except Exception as e:
                print(f"  Groq error: {e}")

        # Fallback to OpenAI
        if self.openai_available:
            try:
                result = await self._call_openai(prompt)
                if result:
                    result['ai_provider'] = 'OpenAI (GPT-3.5-turbo)'
                    result['ai_cost'] = 'Paid'
                    return result
            except Exception as e:
                print(f"  OpenAI error: {e}")

        return {
            'error': 'No AI provider available. Set GROQ_API_KEY or OPENAI_API_KEY in .env',
            'ai_provider': 'None',
            'ai_cost': 'N/A'
        }

    async def _call_groq(self, prompt: str) -> Optional[Dict]:
        """Call Groq API (free tier)"""
        import asyncio
        loop = asyncio.get_event_loop()

        def _sync_call():
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an institutional-grade crypto trading analyst. Provide deep, cross-referenced analysis combining technicals, volume, sentiment, and macro data. Always respond with valid JSON only. No markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2000
            )
            return response.choices[0].message.content

        raw = await loop.run_in_executor(None, _sync_call)
        return self._parse_ai_response(raw)

    async def _call_openai(self, prompt: str) -> Optional[Dict]:
        """Call OpenAI API (fallback)"""
        import asyncio
        loop = asyncio.get_event_loop()

        def _sync_call():
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an institutional-grade crypto trading analyst. Provide deep, cross-referenced analysis combining technicals, volume, sentiment, and macro data. Always respond with valid JSON only. No markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2000
            )
            return response.choices[0].message.content

        raw = await loop.run_in_executor(None, _sync_call)
        return self._parse_ai_response(raw)

    def _parse_ai_response(self, raw: str) -> Optional[Dict]:
        """Parse AI response, handling potential markdown wrapping"""
        if not raw:
            return None

        # Strip markdown code block if present
        text = raw.strip()
        if text.startswith('```'):
            # Remove first line (```json or ```)
            lines = text.split('\n')
            text = '\n'.join(lines[1:])
            if text.endswith('```'):
                text = text[:-3].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass

            return {
                'summary': text[:500],
                'key_insights': ['AI response could not be parsed as JSON'],
                'ai_verdict': 'UNKNOWN',
                'confidence_score': 0
            }


# Global AI service (initialized at startup)
ai_service = None

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def formatPrice(p: float) -> str:
    """Format a price for display"""
    if p < 0.0001:
        return f"${p:.8f}"
    elif p < 0.01:
        return f"${p:.6f}"
    elif p < 1:
        return f"${p:.4f}"
    elif p < 1000:
        return f"${p:.2f}"
    else:
        return f"${p:,.2f}"

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
    version="6.0.0",
    description="Trade-Type-Specific Analysis System with AI-Powered Insights"
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
            coin_dir = os.path.join(model_dir, coin)
            # Try model files in priority order
            model_candidates = [
                'wf_decision_model.pkl',  # Walk-forward validated (best)
                'decision_model.pkl',      # Standard trained
            ]
            feature_candidates = [
                'decision_features.txt',   # Walk-forward features
                'feature_list.txt',        # Standard features
            ]

            loaded = False
            for model_file in model_candidates:
                model_path = os.path.join(coin_dir, model_file)
                if os.path.exists(model_path):
                    try:
                        self.models[coin] = joblib.load(model_path)
                        # Load feature list
                        for feat_file in feature_candidates:
                            feat_path = os.path.join(coin_dir, feat_file)
                            if os.path.exists(feat_path):
                                with open(feat_path, 'r') as f:
                                    self.feature_lists = getattr(self, 'feature_lists', {})
                                    self.feature_lists[coin] = [line.strip() for line in f if line.strip()]
                                break
                        print(f"✅ Loaded model: {coin} ({model_file})")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"❌ Error loading {coin} model ({model_file}): {e}")

            if not loaded:
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
            symbol = coin.replace('_', '')
            resp = requests.get(f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}', timeout=5)
            if resp.status_code == 200:
                return float(resp.json()['price']), 'REST_API'
        except:
            pass
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

    def _fetch_fresh_klines(self, coin: str, interval: str = '1h', limit: int = 168) -> Optional[pd.DataFrame]:
        """Fetch fresh OHLCV klines from Binance API (free, no auth).
        Returns DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            import requests
            symbol = coin.replace('_', '')  # BTC_USDT -> BTCUSDT
            url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"  Binance klines fetch error for {coin}: {e}")
        return None

    def get_volume_profile(self, coin: str, live_price: float = 0, num_levels: int = 20, lookback: int = 168) -> Dict:
        """Calculate Volume Profile (TPO) from fresh Binance data (fallback to CSV).

        Returns POC (Point of Control), Value Area, HVN/LVN levels.
        lookback: number of 1h candles to analyze (default 168 = 7 days)
        num_levels: number of price bins
        live_price: current live price from WebSocket/API
        """
        result = {
            'poc': {'price': 0, 'volume': 0, 'volume_pct': 0},
            'value_area_high': 0,
            'value_area_low': 0,
            'current_price': 0,
            'price_vs_poc': 'AT_POC',
            'levels': [],
            'hvn': [],
            'lvn': [],
            'data_source': 'CSV',
            'analysis': ''
        }

        # Try fresh Binance data first, fall back to CSV
        df = self._fetch_fresh_klines(coin, '1h', lookback)
        if df is not None and len(df) >= 30:
            result['data_source'] = 'LIVE'
        else:
            df = self.load_data(coin)
            if df is not None and len(df) >= 30:
                df = df.tail(min(lookback, len(df)))
                result['data_source'] = 'CSV'
            else:
                result['analysis'] = 'Insufficient data for volume profile'
                return result

        df_vp = df
        # Use live price if provided, otherwise use last close
        current_price = live_price if live_price > 0 else float(df_vp['close'].iloc[-1])
        result['current_price'] = current_price

        try:
            price_min = float(df_vp['low'].min())
            price_max = float(df_vp['high'].max())
            price_range = price_max - price_min

            if price_range <= 0:
                result['analysis'] = 'No price range for volume profile'
                return result

            # Create price bins
            bin_size = price_range / num_levels
            bins = []
            for i in range(num_levels):
                bin_low = price_min + (i * bin_size)
                bin_high = bin_low + bin_size
                bin_mid = (bin_low + bin_high) / 2
                bins.append({
                    'low': bin_low,
                    'high': bin_high,
                    'mid': bin_mid,
                    'volume': 0.0
                })

            # Distribute volume across price bins
            # For each candle, spread its volume across the bins it touches
            for _, row in df_vp.iterrows():
                candle_low = float(row['low'])
                candle_high = float(row['high'])
                candle_vol = float(row['volume'])

                for b in bins:
                    # Calculate overlap between candle range and bin range
                    overlap_low = max(candle_low, b['low'])
                    overlap_high = min(candle_high, b['high'])
                    if overlap_high > overlap_low:
                        candle_range = candle_high - candle_low
                        if candle_range > 0:
                            overlap_pct = (overlap_high - overlap_low) / candle_range
                            b['volume'] += candle_vol * overlap_pct
                        else:
                            # Doji candle - all volume at one price
                            b['volume'] += candle_vol

            # Calculate total volume and percentages
            total_vol = sum(b['volume'] for b in bins)
            if total_vol <= 0:
                result['analysis'] = 'No volume data for profile'
                return result

            # Find POC (bin with highest volume)
            poc_bin = max(bins, key=lambda b: b['volume'])
            result['poc'] = {
                'price': round(poc_bin['mid'], 8),
                'volume': round(poc_bin['volume'], 2),
                'volume_pct': round((poc_bin['volume'] / total_vol) * 100, 1)
            }

            # Build levels list with volume percentages
            max_vol = poc_bin['volume']
            for b in bins:
                vol_pct = (b['volume'] / total_vol) * 100
                vol_relative = (b['volume'] / max_vol) * 100 if max_vol > 0 else 0
                result['levels'].append({
                    'price': round(b['mid'], 8),
                    'price_low': round(b['low'], 8),
                    'price_high': round(b['high'], 8),
                    'volume': round(b['volume'], 2),
                    'volume_pct': round(vol_pct, 1),
                    'volume_relative': round(vol_relative, 1),
                    'is_poc': b is poc_bin
                })

            # Calculate Value Area (70% of volume around POC)
            sorted_bins = sorted(bins, key=lambda b: b['volume'], reverse=True)
            va_vol = 0
            va_target = total_vol * 0.7
            va_bins = []
            for b in sorted_bins:
                va_vol += b['volume']
                va_bins.append(b)
                if va_vol >= va_target:
                    break

            va_prices = [b['mid'] for b in va_bins]
            result['value_area_high'] = round(max(b['high'] for b in va_bins), 8)
            result['value_area_low'] = round(min(b['low'] for b in va_bins), 8)

            # Identify HVN (High Volume Nodes) and LVN (Low Volume Nodes)
            avg_vol = total_vol / num_levels
            for b in bins:
                level_info = {
                    'price': round(b['mid'], 8),
                    'volume_pct': round((b['volume'] / total_vol) * 100, 1)
                }
                if b['volume'] > avg_vol * 1.5:
                    result['hvn'].append(level_info)
                elif b['volume'] < avg_vol * 0.5 and b['volume'] > 0:
                    result['lvn'].append(level_info)

            # Sort HVN/LVN by price
            result['hvn'].sort(key=lambda x: x['price'])
            result['lvn'].sort(key=lambda x: x['price'])

            # Price position relative to POC
            poc_price = poc_bin['mid']
            pct_from_poc = ((current_price - poc_price) / poc_price) * 100
            if pct_from_poc > 2:
                result['price_vs_poc'] = 'ABOVE_POC'
            elif pct_from_poc < -2:
                result['price_vs_poc'] = 'BELOW_POC'
            else:
                result['price_vs_poc'] = 'AT_POC'

            # Build analysis text
            hvn_near = [h for h in result['hvn'] if abs(h['price'] - current_price) / current_price < 0.03]
            lvn_near = [l for l in result['lvn'] if abs(l['price'] - current_price) / current_price < 0.03]

            analysis_parts = []
            analysis_parts.append(f"POC at {formatPrice(poc_price)} ({result['poc']['volume_pct']}% of volume)")
            analysis_parts.append(f"Value Area: {formatPrice(result['value_area_low'])} - {formatPrice(result['value_area_high'])}")

            if result['price_vs_poc'] == 'ABOVE_POC':
                analysis_parts.append('Price ABOVE POC - trading at premium')
            elif result['price_vs_poc'] == 'BELOW_POC':
                analysis_parts.append('Price BELOW POC - trading at discount')
            else:
                analysis_parts.append('Price AT POC - at fair value')

            if hvn_near:
                analysis_parts.append(f'{len(hvn_near)} support/resistance node(s) nearby')
            if lvn_near:
                analysis_parts.append(f'{len(lvn_near)} low-volume gap(s) nearby - fast moves possible')

            result['analysis'] = '. '.join(analysis_parts)

        except Exception as e:
            print(f"Volume profile error for {coin}: {e}")
            result['analysis'] = 'Error calculating volume profile'

        return result

    def get_volume_analysis(self, coin: str, live_price: float = 0) -> Dict:
        """Calculate real-time volume indicators from raw 1h data"""
        result = {
            'obv': {
                'current': 0,
                'trend': 'NEUTRAL',
                'divergence': None
            },
            'mfi': {
                'value': 50.0,
                'zone': 'NEUTRAL',
                'interpretation': ''
            },
            'buy_sell_delta': {
                'delta_pct': 0.0,
                'pressure': 'NEUTRAL',
                'delta_24h': 0.0,
                'strength': 'WEAK'
            },
            'volume_spikes': {
                'current_ratio': 1.0,
                'is_spike': False,
                'spike_count_24h': 0,
                'interpretation': ''
            },
            'force_index': {
                'value': 0.0,
                'trend': 'NEUTRAL'
            },
            'volume_profile': {},
            'overall_signal': 'NEUTRAL',
            'summary': ''
        }

        df = self.load_data(coin)
        if df is None or len(df) < 30:
            result['summary'] = 'Insufficient data for volume analysis'
            return result

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        try:
            # --- OBV ---
            obv_series = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            obv_current = float(obv_series.iloc[-1])
            obv_prev_24 = float(obv_series.iloc[-24]) if len(obv_series) >= 24 else float(obv_series.iloc[0])
            result['obv']['current'] = round(obv_current, 2)

            if obv_current > obv_prev_24 * 1.02:
                result['obv']['trend'] = 'ACCUMULATION'
            elif obv_current < obv_prev_24 * 0.98:
                result['obv']['trend'] = 'DISTRIBUTION'

            # OBV divergence: price vs OBV direction mismatch
            if len(close) >= 24:
                price_change = (float(close.iloc[-1]) - float(close.iloc[-24])) / float(close.iloc[-24])
                obv_change = (obv_current - obv_prev_24) / abs(obv_prev_24) if obv_prev_24 != 0 else 0
                if price_change < -0.01 and obv_change > 0.02:
                    result['obv']['divergence'] = 'BULLISH_DIVERGENCE'
                elif price_change > 0.01 and obv_change < -0.02:
                    result['obv']['divergence'] = 'BEARISH_DIVERGENCE'

            # --- MFI ---
            mfi_value = MFIIndicator(high, low, close, volume, window=14).money_flow_index().iloc[-1]
            if pd.notna(mfi_value):
                result['mfi']['value'] = round(float(mfi_value), 1)
                if mfi_value >= 80:
                    result['mfi']['zone'] = 'OVERBOUGHT'
                    result['mfi']['interpretation'] = 'Overbought - potential reversal or pullback ahead'
                elif mfi_value <= 20:
                    result['mfi']['zone'] = 'OVERSOLD'
                    result['mfi']['interpretation'] = 'Oversold - potential bounce or reversal ahead'
                else:
                    result['mfi']['zone'] = 'NEUTRAL'
                    result['mfi']['interpretation'] = 'Normal money flow - no extreme detected'

            # --- Buy/Sell Volume Delta ---
            df_recent = df.tail(24)
            buy_vol = float(df_recent.loc[df_recent['close'] >= df_recent['open'], 'volume'].sum())
            sell_vol = float(df_recent.loc[df_recent['close'] < df_recent['open'], 'volume'].sum())
            total_vol = buy_vol + sell_vol
            if total_vol > 0:
                delta_pct = ((buy_vol - sell_vol) / total_vol) * 100
                result['buy_sell_delta']['delta_pct'] = round(delta_pct, 1)
                result['buy_sell_delta']['delta_24h'] = round(buy_vol - sell_vol, 2)
                if delta_pct > 20:
                    result['buy_sell_delta']['pressure'] = 'BUYING'
                    result['buy_sell_delta']['strength'] = 'STRONG' if delta_pct > 40 else 'MODERATE'
                elif delta_pct < -20:
                    result['buy_sell_delta']['pressure'] = 'SELLING'
                    result['buy_sell_delta']['strength'] = 'STRONG' if delta_pct < -40 else 'MODERATE'

            # --- Volume Spikes ---
            vol_ma_20 = volume.rolling(20).mean()
            if pd.notna(vol_ma_20.iloc[-1]) and vol_ma_20.iloc[-1] > 0:
                current_ratio = float(volume.iloc[-1]) / float(vol_ma_20.iloc[-1])
                result['volume_spikes']['current_ratio'] = round(current_ratio, 2)
                result['volume_spikes']['is_spike'] = bool(current_ratio > 2.0)
                if len(volume) >= 24:
                    ratios_24 = (volume.tail(24) / vol_ma_20.tail(24)).fillna(1)
                    result['volume_spikes']['spike_count_24h'] = int((ratios_24 > 2.0).sum())
                if current_ratio > 3.0:
                    result['volume_spikes']['interpretation'] = 'Extreme volume spike - significant market event'
                elif current_ratio > 2.0:
                    result['volume_spikes']['interpretation'] = 'Volume spike detected - increased market interest'
                elif current_ratio < 0.5:
                    result['volume_spikes']['interpretation'] = 'Very low volume - lack of conviction'
                else:
                    result['volume_spikes']['interpretation'] = 'Normal volume levels'

            # --- Force Index ---
            fi_value = ForceIndexIndicator(close, volume, window=13).force_index().iloc[-1]
            if pd.notna(fi_value):
                result['force_index']['value'] = round(float(fi_value), 2)
                result['force_index']['trend'] = 'BULLISH' if fi_value > 0 else 'BEARISH'

            # --- Volume Profile ---
            result['volume_profile'] = self.get_volume_profile(coin, live_price=live_price)

            # --- Overall Signal ---
            bullish = 0
            bearish = 0
            if result['obv']['trend'] == 'ACCUMULATION': bullish += 1
            elif result['obv']['trend'] == 'DISTRIBUTION': bearish += 1
            if result['mfi']['zone'] == 'OVERSOLD': bullish += 1
            elif result['mfi']['zone'] == 'OVERBOUGHT': bearish += 1
            if result['buy_sell_delta']['pressure'] == 'BUYING': bullish += 1
            elif result['buy_sell_delta']['pressure'] == 'SELLING': bearish += 1
            if result['force_index']['trend'] == 'BULLISH': bullish += 1
            elif result['force_index']['trend'] == 'BEARISH': bearish += 1
            if result['obv']['divergence'] == 'BULLISH_DIVERGENCE': bullish += 1
            elif result['obv']['divergence'] == 'BEARISH_DIVERGENCE': bearish += 1

            if bullish >= 3:
                result['overall_signal'] = 'BULLISH'
                result['summary'] = f'Volume indicators predominantly bullish ({bullish}/5 bullish signals)'
            elif bearish >= 3:
                result['overall_signal'] = 'BEARISH'
                result['summary'] = f'Volume indicators predominantly bearish ({bearish}/5 bearish signals)'
            else:
                result['overall_signal'] = 'NEUTRAL'
                result['summary'] = f'Volume indicators mixed ({bullish} bullish, {bearish} bearish)'

        except Exception as e:
            print(f"Volume analysis error for {coin}: {e}")
            result['summary'] = f'Error calculating volume indicators'

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

        # Get volume analysis (pass live price for accurate volume profile)
        volume_analysis = self.get_volume_analysis(coin, live_price=price)

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

            # Volume analysis
            'volume_analysis': volume_analysis,

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
    """Start price streaming and AI service on startup"""
    global ai_service
    print("\n" + "=" * 60)
    print("  Crypto AI Trading API v6.0")
    print("=" * 60)
    print("  AI Providers:")
    ai_service = AIService()
    print("=" * 60 + "\n")
    asyncio.create_task(price_streamer.start(engine.coins))

@app.on_event("shutdown")
async def shutdown_event():
    """Stop price streaming on shutdown"""
    price_streamer.stop()

@app.get("/")
async def root():
    # Serve frontend if build exists, otherwise API info
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "build")
    index_path = os.path.join(build_dir, "index.html")
    if os.path.isfile(index_path):
        from starlette.responses import FileResponse
        return FileResponse(index_path)
    return {
        "name": "Crypto AI Trading API",
        "version": "6.0.0",
        "features": [
            "Trade-type-specific analysis",
            "AI-powered insights (Groq + OpenAI)",
            "Paper trading engine",
            "Real-time WebSocket prices",
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

async def _gather_news_data() -> Dict:
    """Helper to gather news data for AI analysis"""
    try:
        crypto_news = await news_service.fetch_crypto_news()
        geo_news = await news_service.fetch_geopolitical_news()
        sentiment = await news_service.get_market_sentiment_summary()
        return {
            'crypto_news': crypto_news,
            'geopolitical_news': geo_news,
            'market_sentiment': sentiment
        }
    except Exception:
        return {}

@app.get("/ai-analysis/{coin}")
async def ai_analysis(
    coin: str,
    capital: float = 1000,
    trade_type: str = "SWING",
    experience: str = "INTERMEDIATE",
    reason: Optional[str] = None,
    recent_losses: int = 0,
    trades_today: int = 0,
    entry_price: Optional[float] = None
):
    """Get AI-powered analysis for a coin (Groq FREE -> OpenAI fallback)"""
    global ai_service
    if ai_service is None:
        ai_service = AIService()

    if not ai_service.groq_available and not ai_service.openai_available:
        raise HTTPException(
            status_code=503,
            detail="No AI provider configured. Set GROQ_API_KEY or OPENAI_API_KEY in .env file."
        )

    try:
        # First run the standard analysis to get technical data
        trade_type_upper = trade_type.upper()
        if trade_type_upper not in ['SCALP', 'SHORT_TERM', 'SWING', 'INVESTMENT']:
            trade_type_upper = 'SWING'

        experience_upper = experience.upper()
        if experience_upper not in ['BEGINNER', 'INTERMEDIATE', 'ADVANCED']:
            experience_upper = 'INTERMEDIATE'

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
        analysis_data = engine.analyze(request)

        # Fetch market sentiment and news in parallel for richer AI context
        coin_upper = coin.upper()
        if '_' not in coin_upper:
            coin_upper = f"{coin_upper}_USDT"

        market_sentiment = None
        news_data = None
        derivatives_data = None
        whale_data = None
        try:
            import asyncio as _aio
            sentiment_task = market_data_service.get_market_sentiment(coin_upper)
            news_task = _gather_news_data()
            derivatives_task = market_data_service.get_derivatives_intelligence(coin_upper)
            whale_task = market_data_service.get_whale_activity(coin_upper)
            results = await _aio.gather(sentiment_task, news_task, derivatives_task, whale_task, return_exceptions=True)
            if not isinstance(results[0], Exception):
                market_sentiment = results[0]
            if not isinstance(results[1], Exception):
                news_data = results[1]
            if not isinstance(results[2], Exception):
                derivatives_data = results[2]
            if not isinstance(results[3], Exception):
                whale_data = results[3]
        except Exception as e:
            print(f"  Warning: Could not fetch sentiment/news/derivatives for AI: {e}")

        # Get AI insights with full context
        ai_result = await ai_service.get_ai_analysis(coin_upper, analysis_data, market_sentiment, news_data, derivatives_data, whale_data)

        return {
            'coin': coin_upper,
            'timestamp': datetime.now().isoformat(),
            'ai_analysis': ai_result,
            'technical_summary': {
                'verdict': analysis_data.get('verdict'),
                'win_probability': analysis_data.get('win_probability'),
                'loss_probability': analysis_data.get('loss_probability'),
                'expectancy': analysis_data.get('expectancy'),
                'price': analysis_data.get('price')
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"  AI analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai-status")
async def ai_status():
    """Check AI provider status"""
    global ai_service
    if ai_service is None:
        ai_service = AIService()
    return {
        'groq_available': ai_service.groq_available,
        'openai_available': ai_service.openai_available,
        'primary_provider': 'Groq (FREE)' if ai_service.groq_available else 'OpenAI' if ai_service.openai_available else 'None',
        'any_available': ai_service.groq_available or ai_service.openai_available
    }

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

        headers = {'User-Agent': 'Mozilla/5.0 (compatible; CryptoAI/6.0)'}
        async with aiohttp.ClientSession(headers=headers) as session:
            for feed_url, source in rss_feeds:
                try:
                    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=15)) as response:
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

        headers = {'User-Agent': 'Mozilla/5.0 (compatible; CryptoAI/6.0)'}
        async with aiohttp.ClientSession(headers=headers) as session:
            for feed_url, source in rss_feeds:
                try:
                    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=15)) as response:
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
# MARKET DATA SERVICE (Fear & Greed, Funding, Open Interest)
# ============================================================

class MarketDataService:
    """Service for fetching on-chain and market sentiment data"""

    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.cache_duration = 300  # 5 minutes
        self.last_fetch: Dict[str, float] = {}

    def _is_cached(self, key: str) -> bool:
        now = datetime.now().timestamp()
        return (key in self.cache and key in self.last_fetch
                and now - self.last_fetch[key] < self.cache_duration)

    def _set_cache(self, key: str, data: Any):
        self.cache[key] = data
        self.last_fetch[key] = datetime.now().timestamp()

    async def get_fear_greed_index(self) -> Dict:
        """Fetch Fear & Greed Index from alternative.me (free, no auth)"""
        cache_key = 'fear_greed'
        if self._is_cached(cache_key):
            return self.cache[cache_key]

        result = {
            'value': 50, 'label': 'Neutral', 'timestamp': None,
            'previous_value': None, 'previous_label': None,
            'trend': 'STABLE', 'source': 'alternative.me'
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://api.alternative.me/fng/?limit=2&format=json',
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        entries = data.get('data', [])
                        if entries:
                            current = entries[0]
                            result['value'] = int(current['value'])
                            result['label'] = current['value_classification']
                            result['timestamp'] = current.get('timestamp')
                            if len(entries) > 1:
                                prev = entries[1]
                                result['previous_value'] = int(prev['value'])
                                result['previous_label'] = prev['value_classification']
                                diff = result['value'] - result['previous_value']
                                result['trend'] = 'IMPROVING' if diff > 5 else 'WORSENING' if diff < -5 else 'STABLE'
        except Exception as e:
            print(f"Fear & Greed API error: {e}")

        self._set_cache(cache_key, result)
        return result

    async def get_funding_rate(self, coin: str) -> Dict:
        """Fetch funding rate from Binance Futures (free, no auth)"""
        symbol = coin.replace('_', '')
        cache_key = f'funding_{symbol}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]

        result = {
            'symbol': symbol, 'funding_rate': 0.0, 'funding_rate_pct': 0.0,
            'sentiment': 'NEUTRAL', 'interpretation': '', 'source': 'Binance Futures'
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1',
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data:
                            rate = float(data[0]['fundingRate'])
                            result['funding_rate'] = round(rate, 6)
                            result['funding_rate_pct'] = round(rate * 100, 4)
                            if rate > 0.001:
                                result['sentiment'] = 'EXTREME_GREED'
                                result['interpretation'] = 'Very high funding - longs paying shorts heavily'
                            elif rate > 0.0003:
                                result['sentiment'] = 'BULLISH'
                                result['interpretation'] = 'Positive funding - market leaning long'
                            elif rate < -0.001:
                                result['sentiment'] = 'EXTREME_FEAR'
                                result['interpretation'] = 'Very negative funding - shorts paying longs'
                            elif rate < -0.0003:
                                result['sentiment'] = 'BEARISH'
                                result['interpretation'] = 'Negative funding - market leaning short'
                            else:
                                result['interpretation'] = 'Neutral funding rate - balanced market'
        except Exception as e:
            print(f"Funding rate API error for {symbol}: {e}")

        self._set_cache(cache_key, result)
        return result

    async def get_open_interest(self, coin: str) -> Dict:
        """Fetch open interest from Binance Futures (free, no auth)"""
        symbol = coin.replace('_', '')
        cache_key = f'oi_{symbol}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]

        result = {
            'symbol': symbol, 'open_interest': 0.0,
            'open_interest_usd': 0.0, 'source': 'Binance Futures'
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}',
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        oi = float(data.get('openInterest', 0))
                        result['open_interest'] = round(oi, 4)
                        async with session.get(
                            f'https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}',
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as price_resp:
                            if price_resp.status == 200:
                                price_data = await price_resp.json()
                                mark_price = float(price_data.get('price', 0))
                                result['open_interest_usd'] = round(oi * mark_price, 2)
        except Exception as e:
            print(f"Open interest API error for {symbol}: {e}")

        self._set_cache(cache_key, result)
        return result

    async def get_long_short_ratio(self, coin: str) -> Dict:
        """Fetch Long/Short ratio for top traders + global accounts from Binance Futures"""
        symbol = coin.replace('_', '')
        cache_key = f'ls_ratio_{symbol}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]

        result = {
            'top_traders': {'long_pct': 50.0, 'short_pct': 50.0, 'ratio': 1.0, 'trend': 'STABLE'},
            'global': {'long_pct': 50.0, 'short_pct': 50.0, 'ratio': 1.0, 'trend': 'STABLE'},
            'signal': 'NEUTRAL',
            'source': 'Binance Futures'
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Top traders long/short ratio
                async with session.get(
                    f'https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={symbol}&period=1h&limit=6',
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and len(data) > 0:
                            latest = data[-1]
                            long_pct = float(latest.get('longAccount', 0.5)) * 100
                            short_pct = float(latest.get('shortAccount', 0.5)) * 100
                            ratio = float(latest.get('longShortRatio', 1.0))
                            result['top_traders'] = {
                                'long_pct': round(long_pct, 1),
                                'short_pct': round(short_pct, 1),
                                'ratio': round(ratio, 3),
                                'trend': 'STABLE'
                            }
                            if len(data) >= 2:
                                old_ratio = float(data[0].get('longShortRatio', 1.0))
                                diff = ratio - old_ratio
                                if diff > 0.05:
                                    result['top_traders']['trend'] = 'LONGS_INCREASING'
                                elif diff < -0.05:
                                    result['top_traders']['trend'] = 'SHORTS_INCREASING'

                # Global accounts long/short ratio
                async with session.get(
                    f'https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=6',
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and len(data) > 0:
                            latest = data[-1]
                            long_pct = float(latest.get('longAccount', 0.5)) * 100
                            short_pct = float(latest.get('shortAccount', 0.5)) * 100
                            ratio = float(latest.get('longShortRatio', 1.0))
                            result['global'] = {
                                'long_pct': round(long_pct, 1),
                                'short_pct': round(short_pct, 1),
                                'ratio': round(ratio, 3),
                                'trend': 'STABLE'
                            }
                            if len(data) >= 2:
                                old_ratio = float(data[0].get('longShortRatio', 1.0))
                                diff = ratio - old_ratio
                                if diff > 0.05:
                                    result['global']['trend'] = 'LONGS_INCREASING'
                                elif diff < -0.05:
                                    result['global']['trend'] = 'SHORTS_INCREASING'

            # Smart money signal: top traders vs crowd divergence
            top_long = result['top_traders']['long_pct']
            global_long = result['global']['long_pct']
            if top_long > 55 and global_long < 45:
                result['signal'] = 'SMART_MONEY_LONG'
            elif top_long < 45 and global_long > 55:
                result['signal'] = 'SMART_MONEY_SHORT'
            elif top_long > 55:
                result['signal'] = 'CONSENSUS_LONG'
            elif top_long < 45:
                result['signal'] = 'CONSENSUS_SHORT'

        except Exception as e:
            print(f"Long/Short ratio API error for {symbol}: {e}")

        self._set_cache(cache_key, result)
        return result

    async def get_taker_volume(self, coin: str) -> Dict:
        """Fetch taker buy/sell volume ratio from Binance Futures"""
        symbol = coin.replace('_', '')
        cache_key = f'taker_{symbol}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]

        result = {
            'buy_vol': 0.0, 'sell_vol': 0.0, 'ratio': 1.0,
            'pressure': 'BALANCED', 'trend': 'STABLE',
            'source': 'Binance Futures'
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={symbol}&period=1h&limit=6',
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and len(data) > 0:
                            latest = data[-1]
                            buy_vol = float(latest.get('buyVol', 0))
                            sell_vol = float(latest.get('sellVol', 0))
                            ratio = float(latest.get('buySellRatio', 1.0))
                            result['buy_vol'] = round(buy_vol, 2)
                            result['sell_vol'] = round(sell_vol, 2)
                            result['ratio'] = round(ratio, 3)

                            if ratio > 1.15:
                                result['pressure'] = 'STRONG_BUYERS'
                            elif ratio > 1.05:
                                result['pressure'] = 'BUYERS'
                            elif ratio < 0.85:
                                result['pressure'] = 'STRONG_SELLERS'
                            elif ratio < 0.95:
                                result['pressure'] = 'SELLERS'

                            if len(data) >= 2:
                                old_ratio = float(data[0].get('buySellRatio', 1.0))
                                diff = ratio - old_ratio
                                if diff > 0.05:
                                    result['trend'] = 'BUYERS_INCREASING'
                                elif diff < -0.05:
                                    result['trend'] = 'SELLERS_INCREASING'
        except Exception as e:
            print(f"Taker volume API error for {symbol}: {e}")

        self._set_cache(cache_key, result)
        return result

    async def get_order_book_depth(self, coin: str) -> Dict:
        """Fetch order book depth from Binance spot API"""
        symbol = coin.replace('_', '')
        cache_key = f'orderbook_{symbol}'
        # Shorter cache for order book (2 min)
        if cache_key in self.cache and cache_key in self.last_fetch:
            if datetime.now().timestamp() - self.last_fetch[cache_key] < 120:
                return self.cache[cache_key]

        result = {
            'total_bid_usd': 0.0, 'total_ask_usd': 0.0,
            'bid_ask_ratio': 1.0, 'imbalance': 'BALANCED',
            'strongest_bid': {'price': 0, 'size_usd': 0},
            'strongest_ask': {'price': 0, 'size_usd': 0},
            'support_level': 0.0, 'resistance_level': 0.0,
            'source': 'Binance'
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20',
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        bids = data.get('bids', [])
                        asks = data.get('asks', [])

                        total_bid_usd = 0.0
                        max_bid_usd = 0.0
                        max_bid_price = 0.0
                        for bid in bids:
                            price = float(bid[0])
                            qty = float(bid[1])
                            usd = price * qty
                            total_bid_usd += usd
                            if usd > max_bid_usd:
                                max_bid_usd = usd
                                max_bid_price = price

                        total_ask_usd = 0.0
                        max_ask_usd = 0.0
                        max_ask_price = 0.0
                        for ask in asks:
                            price = float(ask[0])
                            qty = float(ask[1])
                            usd = price * qty
                            total_ask_usd += usd
                            if usd > max_ask_usd:
                                max_ask_usd = usd
                                max_ask_price = price

                        result['total_bid_usd'] = round(total_bid_usd, 2)
                        result['total_ask_usd'] = round(total_ask_usd, 2)
                        ratio = total_bid_usd / total_ask_usd if total_ask_usd > 0 else 1.0
                        result['bid_ask_ratio'] = round(ratio, 3)

                        if ratio > 1.3:
                            result['imbalance'] = 'STRONG_BID'
                        elif ratio > 1.1:
                            result['imbalance'] = 'BID_HEAVY'
                        elif ratio < 0.7:
                            result['imbalance'] = 'STRONG_ASK'
                        elif ratio < 0.9:
                            result['imbalance'] = 'ASK_HEAVY'

                        result['strongest_bid'] = {'price': max_bid_price, 'size_usd': round(max_bid_usd, 2)}
                        result['strongest_ask'] = {'price': max_ask_price, 'size_usd': round(max_ask_usd, 2)}
                        result['support_level'] = max_bid_price
                        result['resistance_level'] = max_ask_price

        except Exception as e:
            print(f"Order book API error for {symbol}: {e}")

        self._set_cache(cache_key, result)
        return result

    async def get_liquidation_estimates(self, coin: str) -> Dict:
        """Estimate liquidation levels from OI + price + common leverage levels"""
        symbol = coin.replace('_', '')
        cache_key = f'liq_{symbol}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]

        result = {
            'long_liq_levels': [],
            'short_liq_levels': [],
            'recent_signal': 'NONE',
            'oi_change_pct': 0.0,
            'nearest_long_liq': None,
            'nearest_short_liq': None,
            'source': 'Estimated from OI + Leverage'
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Get current price
                current_price = 0.0
                async with session.get(
                    f'https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}',
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        current_price = float(data.get('price', 0))

                if current_price <= 0:
                    self._set_cache(cache_key, result)
                    return result

                # Calculate liquidation levels for common leverage
                leverages = [5, 10, 25, 50, 100]
                for lev in leverages:
                    # Long liquidation: price drops by (1/leverage) * 100%
                    long_liq = current_price * (1 - 1 / lev)
                    dist_pct = (1 / lev) * 100
                    result['long_liq_levels'].append({
                        'leverage': f'{lev}x',
                        'price': round(long_liq, 8),
                        'distance_pct': round(dist_pct, 1)
                    })

                    # Short liquidation: price rises by (1/leverage) * 100%
                    short_liq = current_price * (1 + 1 / lev)
                    result['short_liq_levels'].append({
                        'leverage': f'{lev}x',
                        'price': round(short_liq, 8),
                        'distance_pct': round(dist_pct, 1)
                    })

                result['nearest_long_liq'] = result['long_liq_levels'][-1]  # 100x = closest
                result['nearest_short_liq'] = result['short_liq_levels'][-1]  # 100x = closest

                # Check OI history for recent liquidation signals
                async with session.get(
                    f'https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period=1h&limit=24',
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and len(data) >= 2:
                            latest_oi = float(data[-1].get('sumOpenInterest', 0))
                            oldest_oi = float(data[0].get('sumOpenInterest', 0))
                            if oldest_oi > 0:
                                oi_change = ((latest_oi - oldest_oi) / oldest_oi) * 100
                                result['oi_change_pct'] = round(oi_change, 2)

                                # Large OI drops suggest liquidations
                                if oi_change < -5:
                                    # Check price direction to determine which side got liquidated
                                    latest_price = current_price
                                    # If OI dropped significantly, check recent price move
                                    result['recent_signal'] = 'LONG_SQUEEZE' if oi_change < -10 else 'POSSIBLE_LIQUIDATIONS'
                                elif oi_change > 10:
                                    result['recent_signal'] = 'NEW_POSITIONS_OPENING'

        except Exception as e:
            print(f"Liquidation estimates error for {symbol}: {e}")

        self._set_cache(cache_key, result)
        return result

    async def get_whale_activity(self, coin: str) -> Dict:
        """Detect large transactions / whale activity"""
        symbol = coin.replace('_', '')
        cache_key = f'whale_{symbol}'
        # 10 min cache for whale data (Blockchair rate limits)
        if cache_key in self.cache and cache_key in self.last_fetch:
            if datetime.now().timestamp() - self.last_fetch[cache_key] < 600:
                return self.cache[cache_key]

        result = {
            'recent_large_txs': [],
            'whale_signal': 'NEUTRAL',
            'large_trade_count': 0,
            'avg_large_trade_size': 0.0,
            'alert': None,
            'source': 'Binance Trades'
        }

        try:
            import aiohttp
            base_coin = coin.split('_')[0].upper()

            if base_coin == 'BTC':
                # Blockchair for BTC whale txs (>100 BTC ≈ 10B satoshis)
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        'https://api.blockchair.com/bitcoin/transactions?q=output_total(10000000000..)&limit=5&s=time(desc)',
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            txs = data.get('data', [])
                            for tx in txs[:5]:
                                btc_val = tx.get('output_total', 0) / 1e8
                                result['recent_large_txs'].append({
                                    'hash': tx.get('hash', '')[:16] + '...',
                                    'value': round(btc_val, 2),
                                    'value_display': f'{btc_val:,.2f} BTC',
                                    'time': tx.get('time', ''),
                                    'block': tx.get('block_id', 0)
                                })
                            result['source'] = 'Blockchair'
                            if len(txs) >= 3:
                                result['whale_signal'] = 'ACTIVE'
                                result['alert'] = f'{len(txs)} large BTC transactions detected recently'

            elif base_coin == 'ETH':
                # Blockchair for ETH whale txs (>1000 ETH)
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        'https://api.blockchair.com/ethereum/transactions?q=value(1000000000000000000000..)&limit=5&s=time(desc)',
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            txs = data.get('data', [])
                            for tx in txs[:5]:
                                eth_val = float(tx.get('value', 0)) / 1e18
                                result['recent_large_txs'].append({
                                    'hash': tx.get('hash', '')[:16] + '...',
                                    'value': round(eth_val, 2),
                                    'value_display': f'{eth_val:,.2f} ETH',
                                    'time': tx.get('time', ''),
                                    'block': tx.get('block_id', 0)
                                })
                            result['source'] = 'Blockchair'
                            if len(txs) >= 3:
                                result['whale_signal'] = 'ACTIVE'
                                result['alert'] = f'{len(txs)} large ETH transactions detected recently'

            else:
                # For SOL, PEPE etc: use Binance recent trades to find outliers
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f'https://api.binance.com/api/v3/trades?symbol={symbol}&limit=50',
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status == 200:
                            trades = await resp.json()
                            if trades:
                                sizes = [float(t['quoteQty']) for t in trades]
                                if sizes:
                                    import numpy as np
                                    threshold = np.percentile(sizes, 95)
                                    large_trades = [t for t in trades if float(t['quoteQty']) > threshold]
                                    result['large_trade_count'] = len(large_trades)
                                    if large_trades:
                                        avg_size = sum(float(t['quoteQty']) for t in large_trades) / len(large_trades)
                                        result['avg_large_trade_size'] = round(avg_size, 2)

                                        buy_trades = sum(1 for t in large_trades if not t.get('isBuyerMaker', True))
                                        sell_trades = len(large_trades) - buy_trades

                                        for t in large_trades[:5]:
                                            usd_val = float(t['quoteQty'])
                                            result['recent_large_txs'].append({
                                                'value': round(usd_val, 2),
                                                'value_display': f'${usd_val:,.0f}',
                                                'time': datetime.utcfromtimestamp(t['time'] / 1000).strftime('%H:%M:%S'),
                                                'type': 'BUY' if not t.get('isBuyerMaker', True) else 'SELL'
                                            })

                                        if buy_trades > sell_trades * 2:
                                            result['whale_signal'] = 'ACCUMULATION'
                                            result['alert'] = f'Large buyers dominating ({buy_trades} buys vs {sell_trades} sells)'
                                        elif sell_trades > buy_trades * 2:
                                            result['whale_signal'] = 'DISTRIBUTION'
                                            result['alert'] = f'Large sellers dominating ({sell_trades} sells vs {buy_trades} buys)'
                                        elif len(large_trades) >= 3:
                                            result['whale_signal'] = 'ACTIVE'

        except Exception as e:
            print(f"Whale activity error for {coin}: {e}")

        self._set_cache(cache_key, result)
        return result

    async def get_derivatives_intelligence(self, coin: str) -> Dict:
        """Get combined derivatives data: L/S ratio, taker volume, order book, liquidations"""
        import asyncio as _aio

        tasks = [
            self.get_long_short_ratio(coin),
            self.get_taker_volume(coin),
            self.get_order_book_depth(coin),
            self.get_liquidation_estimates(coin)
        ]
        results = await _aio.gather(*tasks, return_exceptions=True)

        ls_ratio = results[0] if not isinstance(results[0], Exception) else {'signal': 'ERROR'}
        taker = results[1] if not isinstance(results[1], Exception) else {'pressure': 'ERROR'}
        orderbook = results[2] if not isinstance(results[2], Exception) else {'imbalance': 'ERROR'}
        liquidations = results[3] if not isinstance(results[3], Exception) else {'recent_signal': 'ERROR'}

        # Composite signal
        bullish_count = 0
        bearish_count = 0

        ls_signal = ls_ratio.get('signal', 'NEUTRAL')
        if ls_signal in ('SMART_MONEY_LONG', 'CONSENSUS_LONG'):
            bullish_count += 1
        elif ls_signal in ('SMART_MONEY_SHORT', 'CONSENSUS_SHORT'):
            bearish_count += 1

        pressure = taker.get('pressure', 'BALANCED')
        if pressure in ('BUYERS', 'STRONG_BUYERS'):
            bullish_count += 1
        elif pressure in ('SELLERS', 'STRONG_SELLERS'):
            bearish_count += 1

        imbalance = orderbook.get('imbalance', 'BALANCED')
        if imbalance in ('BID_HEAVY', 'STRONG_BID'):
            bullish_count += 1
        elif imbalance in ('ASK_HEAVY', 'STRONG_ASK'):
            bearish_count += 1

        overall = 'BULLISH' if bullish_count >= 2 else 'BEARISH' if bearish_count >= 2 else 'NEUTRAL'

        return {
            'long_short_ratio': ls_ratio,
            'taker_volume': taker,
            'order_book': orderbook,
            'liquidations': liquidations,
            'overall_signal': overall,
            'timestamp': datetime.now().isoformat()
        }

    async def get_market_sentiment(self, coin: str) -> Dict:
        """Get combined market sentiment data for a coin"""
        fear_greed = await self.get_fear_greed_index()
        funding = await self.get_funding_rate(coin)
        oi = await self.get_open_interest(coin)

        signals = []
        fg_value = fear_greed['value']
        if fg_value >= 75: signals.append('BULLISH')
        elif fg_value <= 25: signals.append('BEARISH')
        else: signals.append('NEUTRAL')

        funding_sent = funding['sentiment']
        if funding_sent in ('BULLISH', 'EXTREME_GREED'): signals.append('BULLISH')
        elif funding_sent in ('BEARISH', 'EXTREME_FEAR'): signals.append('BEARISH')
        else: signals.append('NEUTRAL')

        bullish_count = signals.count('BULLISH')
        bearish_count = signals.count('BEARISH')
        overall = 'BULLISH' if bullish_count > bearish_count else 'BEARISH' if bearish_count > bullish_count else 'NEUTRAL'

        return {
            'fear_greed': fear_greed,
            'funding_rate': funding,
            'open_interest': oi,
            'overall_sentiment': overall,
            'timestamp': datetime.now().isoformat()
        }

market_data_service = MarketDataService()

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
# MARKET SENTIMENT API ROUTES
# ============================================================

@app.get("/market-sentiment/{coin}")
async def get_market_sentiment(coin: str):
    """Get market sentiment data (Fear & Greed, Funding Rate, Open Interest)"""
    try:
        coin_upper = coin.upper()
        if '_' not in coin_upper:
            coin_upper = f"{coin_upper}_USDT"
        sentiment = await market_data_service.get_market_sentiment(coin_upper)
        return {
            'coin': coin_upper,
            **sentiment
        }
    except Exception as e:
        print(f"Error fetching market sentiment: {e}")
        return {
            'coin': coin.upper(),
            'error': str(e),
            'fear_greed': {'value': 0, 'label': 'Unknown', 'error': str(e)},
            'funding_rate': {'rate': 0, 'sentiment': 'NEUTRAL', 'error': str(e)},
            'open_interest': {'value': 0, 'error': str(e)},
            'overall_sentiment': 'UNKNOWN',
            'timestamp': datetime.now().isoformat()
        }

@app.get("/fear-greed")
async def get_fear_greed():
    """Get current Fear & Greed Index"""
    try:
        fg = await market_data_service.get_fear_greed_index()
        return fg
    except Exception as e:
        return {'value': 0, 'label': 'Unknown', 'error': str(e)}

@app.get("/klines/{coin}")
async def get_klines(coin: str, interval: str = "1h", limit: int = 168):
    """Get OHLCV kline/candlestick data from Binance (free, no auth)"""
    try:
        coin_upper = coin.upper()
        symbol = coin_upper.replace('_', '')
        if '_' not in coin_upper:
            symbol = f"{coin_upper}USDT"

        valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        if interval not in valid_intervals:
            interval = '1h'
        limit = min(max(limit, 1), 1000)

        import requests as req
        url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
        resp = req.get(url, timeout=10)
        if resp.status_code != 200:
            return {'error': f'Binance API returned {resp.status_code}', 'data': []}

        raw = resp.json()
        data = []
        for k in raw:
            data.append({
                'time': int(k[0]) // 1000,
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })

        return {'coin': coin_upper, 'interval': interval, 'data': data}
    except Exception as e:
        print(f"Klines API error: {e}")
        return {'error': str(e), 'data': []}

@app.get("/derivatives/{coin}")
async def get_derivatives(coin: str):
    """Get derivatives intelligence: L/S ratio, taker volume, order book, liquidation estimates"""
    try:
        coin_upper = coin.upper()
        if '_' not in coin_upper:
            coin_upper = f"{coin_upper}_USDT"
        data = await market_data_service.get_derivatives_intelligence(coin_upper)
        return {'coin': coin_upper, **data}
    except Exception as e:
        print(f"Error fetching derivatives data: {e}")
        return {
            'coin': coin.upper(),
            'error': str(e),
            'overall_signal': 'UNKNOWN',
            'long_short_ratio': {'signal': 'ERROR', 'error': str(e)},
            'taker_volume': {'pressure': 'ERROR', 'error': str(e)},
            'order_book': {'imbalance': 'ERROR', 'error': str(e)},
            'liquidations': {'recent_signal': 'ERROR', 'error': str(e)},
            'timestamp': datetime.now().isoformat()
        }

@app.get("/whales/{coin}")
async def get_whales(coin: str):
    """Get whale activity data for a coin"""
    try:
        coin_upper = coin.upper()
        if '_' not in coin_upper:
            coin_upper = f"{coin_upper}_USDT"
        data = await market_data_service.get_whale_activity(coin_upper)
        return {'coin': coin_upper, **data}
    except Exception as e:
        print(f"Error fetching whale data: {e}")
        return {
            'coin': coin.upper(),
            'error': str(e),
            'whale_signal': 'UNKNOWN',
            'recent_large_txs': [],
            'alert': None
        }

# ============================================================
# BACKTESTING ENDPOINTS
# ============================================================

from backtesting_engine import BacktestingEngine, run_backtest_for_coin, run_backtest_all_coins

# Cache backtest results to avoid re-running expensive computations
_backtest_cache = {}
_backtest_cache_ttl = {}
BACKTEST_CACHE_SECONDS = 300  # 5 minutes

def _get_cached_backtest(cache_key):
    if cache_key in _backtest_cache and cache_key in _backtest_cache_ttl:
        if datetime.now().timestamp() - _backtest_cache_ttl[cache_key] < BACKTEST_CACHE_SECONDS:
            return _backtest_cache[cache_key]
    return None

def _set_cached_backtest(cache_key, result):
    _backtest_cache[cache_key] = result
    _backtest_cache_ttl[cache_key] = datetime.now().timestamp()

@app.get("/backtest/{coin}")
async def run_backtest(coin: str, threshold: float = 0.45, capital: float = 10000):
    """Run backtest for a single coin with configurable parameters."""
    try:
        coin_upper = coin.upper()
        if '_' not in coin_upper:
            coin_upper = f"{coin_upper}_USDT"

        cache_key = f"{coin_upper}_{threshold}_{capital}"
        cached = _get_cached_backtest(cache_key)
        if cached:
            return cached

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, run_backtest_for_coin, coin_upper, threshold, capital
        )

        _set_cached_backtest(cache_key, result)
        return result
    except Exception as e:
        return {'coin': coin.upper(), 'error': str(e)}

@app.get("/backtest")
async def run_backtest_all(threshold: float = 0.45, capital: float = 10000):
    """Run backtest for all supported coins."""
    try:
        cache_key = f"ALL_{threshold}_{capital}"
        cached = _get_cached_backtest(cache_key)
        if cached:
            return cached

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, run_backtest_all_coins, threshold, capital
        )

        # Build summary
        summary = []
        for coin, r in results.items():
            if r.get('metrics'):
                summary.append({
                    'coin': coin,
                    'total_return_pct': r['metrics']['total_return_pct'],
                    'win_rate': r['metrics']['win_rate'],
                    'sharpe_ratio': r['metrics']['sharpe_ratio'],
                    'max_drawdown_pct': r['metrics']['max_drawdown_pct'],
                    'total_trades': r['metrics']['total_trades']
                })

        response = {'results': results, 'summary': summary}
        _set_cached_backtest(cache_key, response)
        return response
    except Exception as e:
        return {'error': str(e)}

@app.get("/backtest/walk-forward/{coin}")
async def run_walk_forward(coin: str, splits: int = 5, threshold: float = 0.45):
    """Run walk-forward validation for a coin."""
    try:
        coin_upper = coin.upper()
        if '_' not in coin_upper:
            coin_upper = f"{coin_upper}_USDT"

        cache_key = f"WF_{coin_upper}_{splits}_{threshold}"
        cached = _get_cached_backtest(cache_key)
        if cached:
            return cached

        engine = BacktestingEngine(coin_upper, threshold=threshold)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, engine.run_walk_forward, splits)

        _set_cached_backtest(cache_key, result)
        return result
    except Exception as e:
        return {'coin': coin.upper(), 'error': str(e)}

@app.get("/backtest/monte-carlo/{coin}")
async def run_monte_carlo(coin: str, simulations: int = 100, threshold: float = 0.45):
    """Run Monte Carlo stress test for a coin (100 randomized simulations)."""
    try:
        coin_upper = coin.upper()
        if '_' not in coin_upper:
            coin_upper = f"{coin_upper}_USDT"

        cache_key = f"MC_{coin_upper}_{simulations}_{threshold}"
        cached = _get_cached_backtest(cache_key)
        if cached:
            return cached

        engine = BacktestingEngine(coin_upper, threshold=threshold)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, engine.run_monte_carlo, simulations)

        _set_cached_backtest(cache_key, result)
        return result
    except Exception as e:
        return {'coin': coin.upper(), 'error': str(e)}

@app.get("/backtest/walk-forward-validation/{coin}")
async def run_walk_forward_validation(coin: str):
    """
    TRUE walk-forward validation with 3-block temporal split.
    Model retrained on TRAIN only, threshold tuned on VALIDATE, single run on TEST.
    No data leakage. No re-optimization.
    """
    try:
        coin_upper = coin.upper()
        if '_' not in coin_upper:
            coin_upper = f"{coin_upper}_USDT"

        cache_key = f"WFV_{coin_upper}"
        cached = _get_cached_backtest(cache_key)
        if cached:
            return cached

        from walk_forward_validation import run_walk_forward
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_walk_forward, coin_upper)

        _set_cached_backtest(cache_key, result)
        return result
    except Exception as e:
        return {'coin': coin.upper(), 'error': str(e)}

@app.get("/backtest/rolling-robustness/{coin}")
async def run_rolling_robustness(coin: str):
    """
    Rolling walk-forward robustness validation.
    Tests strategy across multiple market regimes with expanding training windows.
    """
    try:
        coin_upper = coin.upper()
        if '_' not in coin_upper:
            coin_upper = f"{coin_upper}_USDT"

        cache_key = f"ROLLING_{coin_upper}"
        cached = _get_cached_backtest(cache_key)
        if cached:
            return cached

        from rolling_walk_forward import run_coin_robustness
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_coin_robustness, coin_upper)

        _set_cached_backtest(cache_key, result)
        return result
    except Exception as e:
        return {'coin': coin.upper(), 'error': str(e)}

@app.get("/backtest/rolling-robustness")
async def run_all_rolling_robustness():
    """Rolling robustness for all coins at once."""
    try:
        cache_key = "ROLLING_ALL"
        cached = _get_cached_backtest(cache_key)
        if cached:
            return cached

        from rolling_walk_forward import run_all
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_all)

        _set_cached_backtest(cache_key, result)
        return result
    except Exception as e:
        return {'error': str(e)}

# ============================================================
# PAPER TRADING ENDPOINTS
# ============================================================

from paper_trader import paper_trader

@app.on_event("startup")
async def auto_start_paper_trading():
    """Auto-start paper trading if AUTO_START_PAPER_TRADING env var is set."""
    auto_start = os.getenv('AUTO_START_PAPER_TRADING', '').lower()
    if auto_start in ('1', 'true', 'yes'):
        capital = float(os.getenv('PAPER_TRADING_CAPITAL', '10000'))
        print(f"\n  [PAPER] Auto-starting paper trading (capital=${capital:,.0f})...")
        paper_trader.start(capital=capital)

@app.post("/paper-trading/start")
async def start_paper_trading(capital: float = None):
    """Start the live paper trading bot. Optional capital param sets initial equity."""
    try:
        result = paper_trader.start(capital=capital)
        return result
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

@app.post("/paper-trading/stop")
async def stop_paper_trading():
    """Stop the paper trading bot."""
    try:
        result = paper_trader.stop()
        return result
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

@app.post("/paper-trading/reset")
async def reset_paper_trading():
    """Stop and reset paper trading state (clears all trades and equity)."""
    try:
        paper_trader.stop()
        import os as _os
        from paper_trader import STATE_FILE
        if _os.path.exists(STATE_FILE):
            _os.remove(STATE_FILE)
        paper_trader.state = {}
        return {'status': 'reset', 'message': 'State cleared. Ready to start fresh.'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

@app.get("/paper-trading/status")
async def get_paper_trading_status():
    """Get current paper trading status, equity, and open positions."""
    try:
        return paper_trader.get_status()
    except Exception as e:
        return {'running': False, 'error': str(e)}

@app.get("/paper-trading/trades")
async def get_paper_trading_trades():
    """Get full paper trading trade log."""
    try:
        return {'trades': paper_trader.get_trades()}
    except Exception as e:
        return {'trades': [], 'error': str(e)}

@app.get("/paper-trading/metrics")
async def get_paper_trading_metrics():
    """Get paper trading metrics: WR, PF, Sharpe, DD, per-coin breakdown."""
    try:
        return paper_trader.get_metrics()
    except Exception as e:
        return {'error': str(e), 'total_trades': 0}

# ============================================================
# SERVE FRONTEND (production static files)
# ============================================================

FRONTEND_BUILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "build")

if os.path.isdir(FRONTEND_BUILD):
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import FileResponse

    # Serve static assets (JS, CSS, images)
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_BUILD, "static")), name="static")

    # Catch-all: serve index.html for any non-API route (React Router)
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Don't intercept API routes
        file_path = os.path.join(FRONTEND_BUILD, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_BUILD, "index.html"))

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Crypto AI Trading API v6.0")
    print("Trade-Type-Specific Analysis + AI-Powered Insights")
    print("Volume Analysis + Market Sentiment Enabled")
    print("News Feed Service Enabled")
    print("Backtesting Engine Enabled")
    print("Paper Trading Engine Available")
    if os.path.isdir(FRONTEND_BUILD):
        print("Frontend Dashboard: Serving from build/")
    uvicorn.run(app, host="0.0.0.0", port=8000)