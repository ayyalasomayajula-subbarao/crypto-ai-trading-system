"""
Precision Verdict System - Fusion Layer
========================================
Combines ML model + 13 signals into one structured verdict with confidence,
agreement scoring, AI reasoning, caching, and accuracy tracking.

Architecture:
  Layer 1 (existing): engine.get_model_probabilities()  → ML signal #1
  Layer 2 (existing): feature CSV                        → RSI, MACD, ADX, SMAs
  Layer 3 (existing): market_data_service               → funding, OI, L/S, OB, F&G
  Layer 4 (existing): engine.get_btc_context()          → BTC macro signal
  Layer 5 (NEW):      VerdictEngine                     → fuse + score + AI reasoning

Signal architecture (13 signals):
  # | Name              | Category  | Weight | Source
  1 | ML Model P(UP/DN) | MODEL     | 3.0    | LightGBM (WF-validated)
  2 | Price Trend       | TREND     | 2.0    | feature CSV (dist_sma_21/50)
  3 | Trend Strength    | TREND     | 1.5    | feature CSV (adx)
  4 | RSI               | MOMENTUM  | 1.0    | feature CSV (rsi)
  5 | MACD diff         | MOMENTUM  | 1.0    | feature CSV (macd_diff)
  6 | Volume Delta      | VOLUME    | 1.5    | feature CSV (imbalance_4h_ma)
  7 | Funding Rate      | SENTIMENT | 1.5    | Binance Futures (live)
  8 | Open Interest Δ   | FLOW      | 1.0    | Binance Futures (live)
  9 | Long/Short Ratio  | FLOW      | 1.0    | Binance Futures (live)
 10 | Order Book        | FLOW      | 1.0    | Binance Futures (live)
 11 | Fear & Greed      | SENTIMENT | 0.5    | alternative.me (live)
 12 | BTC Context       | MACRO     | 1.5    | engine.get_btc_context()
 13 | Regime Gate       | MACRO     | 1.5    | feature CSV (1w_dist_sma_50)

Max possible score: 36.0 (all STRONG_BULLISH × weights)
"""

import os
import json
import sqlite3
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# ─── WF-validated coin statuses (frozen — do not change without re-running WF) ─
COIN_MODEL_STATUS = {
    'BTC_USDT':  {'status': 'MARGINAL',    'threshold': 0.50},
    'ETH_USDT':  {'status': 'NOT_VIABLE',  'threshold': None},
    'SOL_USDT':  {'status': 'NOT_VIABLE',  'threshold': None},
    'PEPE_USDT': {'status': 'NOT_VIABLE',  'threshold': None},
    'AVAX_USDT': {'status': 'NOT_VIABLE',  'threshold': None},
    'BNB_USDT':  {'status': 'NOT_VIABLE',  'threshold': None},
    'LINK_USDT': {'status': 'MARGINAL',    'threshold': 0.55},
    'ARB_USDT':  {'status': 'NOT_VIABLE',  'threshold': None},
    'OP_USDT':   {'status': 'NOT_VIABLE',  'threshold': None},
    'INJ_USDT':  {'status': 'NOT_VIABLE',  'threshold': None},
}

# ─── Coin-specific TP/SL from CLAUDE.md ────────────────────────────────────────
COIN_TP_SL = {
    'BTC_USDT':  {'tp_pct': 3.0,  'sl_pct': 1.5, 'max_hold_h': 48},
    'ETH_USDT':  {'tp_pct': 4.5,  'sl_pct': 1.5, 'max_hold_h': 48},
    'SOL_USDT':  {'tp_pct': 7.5,  'sl_pct': 2.5, 'max_hold_h': 72},
    'PEPE_USDT': {'tp_pct': 15.0, 'sl_pct': 5.0, 'max_hold_h': 48},
    'AVAX_USDT': {'tp_pct': 7.5,  'sl_pct': 2.5, 'max_hold_h': 72},
    'BNB_USDT':  {'tp_pct': 6.0,  'sl_pct': 2.0, 'max_hold_h': 48},
    'LINK_USDT': {'tp_pct': 7.5,  'sl_pct': 2.5, 'max_hold_h': 72},
    'ARB_USDT':  {'tp_pct': 8.0,  'sl_pct': 3.0, 'max_hold_h': 72},
    'OP_USDT':   {'tp_pct': 8.0,  'sl_pct': 3.0, 'max_hold_h': 72},
    'INJ_USDT':  {'tp_pct': 8.0,  'sl_pct': 3.0, 'max_hold_h': 72},
}

TRADE_TYPE_TP_SL = {
    'SCALP':      {'tp_pct': 3.0,  'sl_pct': 1.5},
    'SHORT_TERM': {'tp_pct': 5.0,  'sl_pct': 2.5},
    'SWING':      {'tp_pct': 10.0, 'sl_pct': 4.0},
    'INVESTMENT': {'tp_pct': 25.0, 'sl_pct': 8.0},
}

# ─── Position sizing: % of capital to risk ─────────────────────────────────────
TRADE_TYPE_POSITION_PCT = {
    'SCALP': 3.0, 'SHORT_TERM': 5.0, 'SWING': 5.0, 'INVESTMENT': 10.0
}

# ─── Regime gate: which feature column per coin ────────────────────────────────
REGIME_GATE_COL = {
    'BTC_USDT':  '1w_dist_sma_50',
    'ETH_USDT':  '1w_dist_sma_50',
    'SOL_USDT':  '1w_dist_sma_50',
    'PEPE_USDT': '1d_dist_sma_50',
    'AVAX_USDT': '1w_dist_sma_50',
    'BNB_USDT':  '1w_dist_sma_50',
    'LINK_USDT': '1w_dist_sma_50',
    'ARB_USDT':  '1w_dist_sma_50',
    'OP_USDT':   '1w_dist_sma_50',
    'INJ_USDT':  '1w_dist_sma_50',
}

# ─── Signal score constants ─────────────────────────────────────────────────────
STRONG_BULLISH =  2
BULLISH        =  1
NEUTRAL        =  0
BEARISH        = -1
STRONG_BEARISH = -2

SIGNAL_EMOJI = {
    STRONG_BULLISH: '🟢',
    BULLISH:        '🟢',
    NEUTRAL:        '⚪',
    BEARISH:        '🔴',
    STRONG_BEARISH: '🔴',
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class SignalFactor:
    def __init__(self, name: str, category: str, raw_value: Any,
                 score: int, weight: float, reason: str):
        self.name      = name
        self.category  = category
        self.raw_value = raw_value
        self.score     = score        # STRONG_BULLISH…STRONG_BEARISH
        self.weight    = weight
        self.reason    = reason

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight

    def to_dict(self) -> Dict:
        return {
            'name':      self.name,
            'category':  self.category,
            'value':     str(self.raw_value),
            'score':     self.score,
            'emoji':     SIGNAL_EMOJI.get(self.score, '⚪'),
            'weight':    self.weight,
            'reason':    self.reason,
            'w_score':   round(self.weighted_score, 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
# VERDICT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class VerdictEngine:
    """
    Fuses existing TradingEngine + MarketDataService outputs into a
    single structured verdict. No duplicate API calls.
    """

    def __init__(self, trading_engine, market_data_service):
        self.engine = trading_engine
        self.mds    = market_data_service
        self._cache: Dict[str, Dict] = {}   # {coin: {data, ts}}
        self._init_db()
        self._init_ai()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_db(self):
        """Create verdict_history table in existing agent_memory.db"""
        try:
            conn = sqlite3.connect('data/agent_memory.db')
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verdict_history (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin             TEXT    NOT NULL,
                    timestamp        TEXT    NOT NULL,
                    verdict          TEXT    NOT NULL,
                    confidence       INTEGER,
                    consensus_score  REAL,
                    price_at_verdict REAL,
                    trade_type       TEXT,
                    price_after_24h  REAL,
                    price_after_48h  REAL,
                    price_after_7d   REAL,
                    was_correct      INTEGER,
                    checked_at       TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"VerdictEngine DB init error: {e}")

    def _init_ai(self):
        """Initialize Groq client for AI reasoning"""
        self.groq_client = None
        groq_key = os.getenv('GROQ_API_KEY', '')
        if groq_key and groq_key != 'gsk_your_groq_key_here':
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
            except Exception:
                pass

    # ── Main entry point ──────────────────────────────────────────────────────

    async def generate(self, coin: str, trade_type: str = 'SWING',
                       capital: float = 1000) -> Dict:
        """
        Generate a precision verdict for a coin.
        Cached for 5 minutes per coin.
        """
        coin_upper = coin.upper()
        if '_' not in coin_upper:
            coin_upper = f"{coin_upper}_USDT"

        trade_type = trade_type.upper()
        if trade_type not in TRADE_TYPE_TP_SL:
            trade_type = 'SWING'

        # ── Cache check ──
        cached = self._cache.get(coin_upper)
        if cached and time.time() - cached['ts'] < 300:
            result = dict(cached['data'])
            result['cached'] = True
            result['trade_setup'] = self._calculate_trade_setup(
                cached['data']['price'], cached['data']['verdict'],
                coin_upper, trade_type, capital
            )
            return result

        # ── Check for old verdicts needing accuracy update (background-safe) ──
        self._check_accuracy_async(coin_upper)

        # ── Get live price ──
        price, _ = self.engine.get_live_price(coin_upper)

        # ── Load feature CSV once ──
        df = self.engine.load_data(coin_upper, multi_tf=True)

        # ── Score all 13 signals ──
        signals = await self._score_all_signals(coin_upper, df, price)

        # ── Calculate verdict from signals ──
        model_status = COIN_MODEL_STATUS.get(coin_upper, {}).get('status', 'NO_MODEL')
        verdict_data = self._calculate_verdict(signals, model_status)

        # ── Get AI reasoning ──
        ai = await self._get_ai_reasoning(coin_upper, price, verdict_data, signals)

        # ── Trade setup ──
        trade_setup = self._calculate_trade_setup(
            price, verdict_data['verdict'], coin_upper, trade_type, capital
        )

        result = {
            'coin':            coin_upper,
            'timestamp':       datetime.now().isoformat(),
            'price':           price,
            'verdict':         verdict_data['verdict'],
            'confidence':      verdict_data['confidence'],
            'consensus_score': round(verdict_data['score'], 2),
            'max_score':       verdict_data['max_score'],
            'bullish_count':   verdict_data['bullish_count'],
            'bearish_count':   verdict_data['bearish_count'],
            'neutral_count':   verdict_data['neutral_count'],
            'agreement_pct':   verdict_data['agreement_pct'],
            'agreement_label': verdict_data['agreement_label'],
            'model_status':    model_status,
            'signals':         [s.to_dict() for s in signals],
            'summary':         ai.get('summary', ''),
            'reasons':         ai.get('reasons', []),
            'risks':           ai.get('risks', []),
            'watch_for':       ai.get('watch_for', []),
            'trade_setup':     trade_setup,
            'cached':          False,
        }

        # ── Store in cache + DB ──
        self._cache[coin_upper] = {'data': result, 'ts': time.time()}
        self._store_verdict(coin_upper, result, trade_type)

        return result

    # ── Signal scoring ────────────────────────────────────────────────────────

    async def _score_all_signals(self, coin: str, df, price: float) -> List[SignalFactor]:
        """Score all 13 signals. Returns list of SignalFactor objects."""
        signals = []

        # Read latest feature row once
        feat = {}
        if df is not None and len(df) > 0:
            row = df.iloc[-1]
            feat = row.to_dict()

        # ── Fetch live derivatives data in parallel ──
        try:
            deriv_task = self.mds.get_derivatives_intelligence(coin)
            sent_task  = self.mds.get_market_sentiment(coin)
            deriv, sentiment = await asyncio.gather(deriv_task, sent_task,
                                                     return_exceptions=True)
            if isinstance(deriv, Exception):
                deriv = {}
            if isinstance(sentiment, Exception):
                sentiment = {}
        except Exception:
            deriv = {}
            sentiment = {}

        # ── Signal #1: ML Model ───────────────────────────────────────────────
        signals.append(self._score_ml_model(coin, feat))

        # ── Signal #2: Price Trend (SMA20/50) ────────────────────────────────
        signals.append(self._score_price_trend(feat))

        # ── Signal #3: Trend Strength (ADX) ──────────────────────────────────
        signals.append(self._score_adx(feat))

        # ── Signal #4: RSI ────────────────────────────────────────────────────
        signals.append(self._score_rsi(feat))

        # ── Signal #5: MACD diff ──────────────────────────────────────────────
        signals.append(self._score_macd(feat))

        # ── Signal #6: Volume Delta ───────────────────────────────────────────
        signals.append(self._score_volume_delta(feat))

        # ── Signal #7: Funding Rate ───────────────────────────────────────────
        funding = sentiment.get('funding_rate', {}) if isinstance(sentiment, dict) else {}
        signals.append(self._score_funding(funding))

        # ── Signal #8: Open Interest Δ ────────────────────────────────────────
        liq = deriv.get('liquidations', {}) if isinstance(deriv, dict) else {}
        signals.append(self._score_open_interest(liq))

        # ── Signal #9: Long/Short Ratio ───────────────────────────────────────
        ls = deriv.get('long_short_ratio', {}) if isinstance(deriv, dict) else {}
        signals.append(self._score_long_short(ls))

        # ── Signal #10: Order Book ────────────────────────────────────────────
        ob = deriv.get('order_book', {}) if isinstance(deriv, dict) else {}
        signals.append(self._score_order_book(ob))

        # ── Signal #11: Fear & Greed ──────────────────────────────────────────
        fg = sentiment.get('fear_greed', {}) if isinstance(sentiment, dict) else {}
        signals.append(self._score_fear_greed(fg))

        # ── Signal #12: BTC Context ───────────────────────────────────────────
        signals.append(self._score_btc_context(coin))

        # ── Signal #13: Regime Gate (weekly SMA50) ────────────────────────────
        signals.append(self._score_regime_gate(coin, feat))

        return signals

    # ── Individual signal scorers ─────────────────────────────────────────────

    def _score_ml_model(self, coin: str, feat: Dict) -> SignalFactor:
        """Signal #1: ML Model P(UP) vs P(DOWN) — highest weight 3.0"""
        info = COIN_MODEL_STATUS.get(coin, {})
        status = info.get('status', 'NO_MODEL')
        threshold = info.get('threshold', None)

        probs = self.engine.get_model_probabilities(coin)
        win_p  = probs.get('win', 0)
        loss_p = probs.get('loss', 0)
        err    = probs.get('error')

        if err or status == 'NO_MODEL' or coin not in self.engine.models:
            return SignalFactor(
                'ML Model', 'MODEL', 'NO_MODEL',
                NEUTRAL, 3.0,
                'No validated model available — signal skipped'
            )

        if status == 'NOT_VIABLE':
            return SignalFactor(
                'ML Model', 'MODEL', f'P(UP)={win_p:.1f}% P(DN)={loss_p:.1f}%',
                NEUTRAL, 3.0,
                f'Model NOT_VIABLE in walk-forward validation — not trusted'
            )

        # MARGINAL model — use validated threshold
        thresh_pct = (threshold or 0.50) * 100
        edge = abs(win_p - loss_p)

        if win_p >= thresh_pct and win_p >= loss_p:
            score = STRONG_BULLISH if edge > 10 else BULLISH
            reason = (f'P(UP)={win_p:.1f}% ≥ WF threshold {thresh_pct:.0f}%, '
                      f'edge +{edge:.1f}% — model says LONG')
        elif loss_p >= thresh_pct and loss_p > win_p:
            score = STRONG_BEARISH if edge > 10 else BEARISH
            reason = (f'P(DOWN)={loss_p:.1f}% ≥ WF threshold {thresh_pct:.0f}%, '
                      f'edge +{edge:.1f}% — model says SHORT')
        elif loss_p > 50:
            score = BEARISH
            reason = f'P(DOWN)={loss_p:.1f}% > 50% — model leans bearish'
        else:
            score = NEUTRAL
            reason = (f'P(UP)={win_p:.1f}% < threshold {thresh_pct:.0f}%'
                      f' — model says WAIT')

        return SignalFactor(
            'ML Model', 'MODEL',
            f'P(UP)={win_p:.1f}% P(DN)={loss_p:.1f}% [{status}]',
            score, 3.0, reason
        )

    def _score_price_trend(self, feat: Dict) -> SignalFactor:
        """Signal #2: Price vs SMA21 and SMA50 — weight 2.0"""
        d21 = float(feat.get('1h_dist_sma_21', 0) or 0)
        d50 = float(feat.get('1h_dist_sma_50', 0) or 0)

        if d21 > 1.0 and d50 > 1.0:
            score = STRONG_BULLISH if d21 > 3 else BULLISH
            reason = f'Price +{d21:.1f}% above SMA21, +{d50:.1f}% above SMA50 — uptrend'
        elif d21 > 0 and d50 < 0:
            score = NEUTRAL
            reason = f'Price above SMA21 (+{d21:.1f}%) but below SMA50 ({d50:.1f}%) — mixed'
        elif d21 < -1.0 and d50 < -1.0:
            score = STRONG_BEARISH if d21 < -3 else BEARISH
            reason = f'Price {d21:.1f}% below SMA21, {d50:.1f}% below SMA50 — downtrend'
        elif d21 < 0 and d50 > 0:
            score = NEUTRAL
            reason = f'Price below SMA21 ({d21:.1f}%) but above SMA50 (+{d50:.1f}%) — consolidation'
        else:
            score = NEUTRAL
            reason = f'Price near SMA21 ({d21:.1f}%) — sideways'

        return SignalFactor('Price Trend', 'TREND',
                            f'SMA21 dist:{d21:+.1f}% SMA50 dist:{d50:+.1f}%',
                            score, 2.0, reason)

    def _score_adx(self, feat: Dict) -> SignalFactor:
        """Signal #3: ADX trend strength — weight 1.5"""
        adx   = float(feat.get('1h_adx', 0) or 0)
        d21   = float(feat.get('1h_dist_sma_21', 0) or 0)
        trend_direction = 'bullish' if d21 >= 0 else 'bearish'

        if adx >= 30:
            score = STRONG_BULLISH if d21 >= 0 else STRONG_BEARISH
            reason = f'ADX {adx:.0f} — strong {trend_direction} trend, momentum is real'
        elif adx >= 20:
            score = BULLISH if d21 >= 0 else BEARISH
            reason = f'ADX {adx:.0f} — moderate {trend_direction} trend, tradeable'
        elif adx >= 10:
            score = NEUTRAL
            reason = f'ADX {adx:.0f} — weak trend, range-bound conditions'
        else:
            score = NEUTRAL
            reason = f'ADX {adx:.0f} — no trend detected, wait for breakout'

        return SignalFactor('Trend Strength (ADX)', 'TREND',
                            f'ADX={adx:.0f}', score, 1.5, reason)

    def _score_rsi(self, feat: Dict) -> SignalFactor:
        """Signal #4: RSI — weight 1.0"""
        rsi = float(feat.get('1h_rsi', 50) or 50)

        if rsi <= 25:
            score = STRONG_BULLISH
            reason = f'RSI {rsi:.0f} — extremely oversold, strong reversal potential'
        elif rsi <= 35:
            score = BULLISH
            reason = f'RSI {rsi:.0f} — oversold, look for bullish reversal'
        elif rsi >= 75:
            score = STRONG_BEARISH
            reason = f'RSI {rsi:.0f} — extremely overbought, pullback very likely'
        elif rsi >= 65:
            score = BEARISH
            reason = f'RSI {rsi:.0f} — overbought, momentum exhaustion risk'
        elif rsi >= 55:
            score = BULLISH
            reason = f'RSI {rsi:.0f} — healthy bullish momentum'
        elif rsi <= 45:
            score = BEARISH
            reason = f'RSI {rsi:.0f} — weak momentum, sellers in control'
        else:
            score = NEUTRAL
            reason = f'RSI {rsi:.0f} — neutral, no directional edge'

        return SignalFactor('RSI', 'MOMENTUM', f'{rsi:.0f}', score, 1.0, reason)

    def _score_macd(self, feat: Dict) -> SignalFactor:
        """Signal #5: MACD diff — weight 1.0"""
        macd = float(feat.get('1h_macd_diff', 0) or 0)
        # Normalize to determine magnitude (relative to itself)
        if macd > 0:
            score = STRONG_BULLISH if macd > abs(macd) * 2 else BULLISH
            reason = f'MACD diff +{macd:.4f} — bullish momentum, histogram positive'
        elif macd < 0:
            score = STRONG_BEARISH if abs(macd) > abs(macd) * 2 else BEARISH
            reason = f'MACD diff {macd:.4f} — bearish momentum, histogram negative'
        else:
            score = NEUTRAL
            reason = 'MACD diff ≈ 0 — no momentum signal'

        # Use the sign for clear scoring
        if macd > 0.001:
            score = BULLISH
            reason = f'MACD diff +{macd:.4f} — histogram positive, bullish momentum'
        elif macd > 0:
            score = NEUTRAL
            reason = f'MACD diff +{macd:.4f} — barely positive'
        elif macd < -0.001:
            score = BEARISH
            reason = f'MACD diff {macd:.4f} — histogram negative, bearish momentum'
        else:
            score = NEUTRAL
            reason = f'MACD diff {macd:.4f} — near zero'

        return SignalFactor('MACD Diff', 'MOMENTUM', f'{macd:.5f}',
                            score, 1.0, reason)

    def _score_volume_delta(self, feat: Dict) -> SignalFactor:
        """Signal #6: Buy/sell imbalance from taker volume — weight 1.5"""
        # imbalance_4h_ma: signed value, +1 = pure buying, -1 = pure selling
        imb = float(feat.get('imbalance_4h_ma', 0) or 0)
        pct = imb * 100  # convert to percentage

        if pct > 15:
            score = STRONG_BULLISH
            reason = f'Buy imbalance +{pct:.1f}% (4h avg) — strong buy pressure'
        elif pct > 5:
            score = BULLISH
            reason = f'Buy imbalance +{pct:.1f}% (4h avg) — buyers in control'
        elif pct < -15:
            score = STRONG_BEARISH
            reason = f'Sell imbalance {pct:.1f}% (4h avg) — strong sell pressure'
        elif pct < -5:
            score = BEARISH
            reason = f'Sell imbalance {pct:.1f}% (4h avg) — sellers in control'
        else:
            score = NEUTRAL
            reason = f'Volume imbalance {pct:.1f}% — balanced buying/selling'

        return SignalFactor('Volume Delta', 'VOLUME',
                            f'imbalance_4h={pct:.1f}%', score, 1.5, reason)

    def _score_funding(self, funding: Dict) -> SignalFactor:
        """Signal #7: Funding rate sentiment — weight 1.5"""
        rate_pct = float(funding.get('funding_rate_pct', 0) or 0)
        sentiment = funding.get('sentiment', 'NEUTRAL')
        interp    = funding.get('interpretation', '')

        if sentiment == 'EXTREME_GREED':
            score = STRONG_BEARISH
            reason = f'Funding {rate_pct:.4f}% — extreme longs, squeeze risk high'
        elif sentiment == 'BULLISH':
            score = BEARISH
            reason = f'Funding {rate_pct:.4f}% — longs paying, over-leveraged bulls'
        elif sentiment == 'EXTREME_FEAR':
            score = STRONG_BULLISH
            reason = f'Funding {rate_pct:.4f}% — shorts paying, squeeze squeeze risk'
        elif sentiment == 'BEARISH':
            score = BULLISH
            reason = f'Funding {rate_pct:.4f}% — shorts paying longs, contrarian bullish'
        else:
            score = NEUTRAL
            reason = f'Funding {rate_pct:.4f}% — balanced, no leverage bias'

        return SignalFactor('Funding Rate', 'SENTIMENT',
                            f'{rate_pct:.4f}% ({sentiment})', score, 1.5, reason)

    def _score_open_interest(self, liq: Dict) -> SignalFactor:
        """Signal #8: Open interest 24h change — weight 1.0"""
        oi_chg = float(liq.get('oi_change_pct', 0) or 0)
        oi_sig = liq.get('recent_signal', 'NONE')

        if oi_sig == 'LONG_SQUEEZE':
            score = BEARISH
            reason = f'OI signal: LONG_SQUEEZE — longs being liquidated'
        elif oi_sig == 'SHORT_SQUEEZE':
            score = BULLISH
            reason = f'OI signal: SHORT_SQUEEZE — shorts being squeezed'
        elif oi_chg >= 5:
            score = BULLISH
            reason = f'OI +{oi_chg:.1f}% (24h) — new money entering, conviction rising'
        elif oi_chg <= -5:
            score = BEARISH
            reason = f'OI {oi_chg:.1f}% (24h) — positions closing, conviction falling'
        else:
            score = NEUTRAL
            reason = f'OI {oi_chg:+.1f}% (24h) — stable positioning'

        return SignalFactor('Open Interest Δ', 'FLOW',
                            f'{oi_chg:+.1f}% ({oi_sig})', score, 1.0, reason)

    def _score_long_short(self, ls: Dict) -> SignalFactor:
        """Signal #9: Long/Short ratio (smart money) — weight 1.0"""
        signal = ls.get('signal', 'NEUTRAL')
        top = ls.get('top_traders', {})
        top_long = float(top.get('long_pct', 50) or 50)

        if signal == 'SMART_MONEY_LONG':
            score = STRONG_BULLISH
            reason = (f'Smart money {top_long:.0f}% long vs crowd short — '
                      'institutional longs, contrarian crowd short')
        elif signal == 'SMART_MONEY_SHORT':
            score = STRONG_BEARISH
            reason = (f'Smart money {top_long:.0f}% long vs crowd long — '
                      'institutional shorts vs retail longs')
        elif signal == 'CONSENSUS_LONG':
            score = BULLISH
            reason = f'Top traders {top_long:.0f}% long — consensus bullish'
        elif signal == 'CONSENSUS_SHORT':
            score = BEARISH
            reason = f'Top traders {top_long:.0f}% long — consensus bearish'
        else:
            score = NEUTRAL
            reason = f'L/S ratio neutral ({top_long:.0f}% top traders long)'

        return SignalFactor('Long/Short Ratio', 'FLOW',
                            f'{top_long:.0f}% long ({signal})', score, 1.0, reason)

    def _score_order_book(self, ob: Dict) -> SignalFactor:
        """Signal #10: Order book bid/ask imbalance — weight 1.0"""
        imbalance = ob.get('imbalance', 'BALANCED')
        ratio = float(ob.get('bid_ask_ratio', 1.0) or 1.0)

        if imbalance == 'STRONG_BID':
            score = STRONG_BULLISH
            reason = f'Order book STRONG BID (ratio {ratio:.2f}) — heavy buy wall'
        elif imbalance == 'BID_HEAVY':
            score = BULLISH
            reason = f'Order book BID heavy (ratio {ratio:.2f}) — more buyers than sellers'
        elif imbalance == 'STRONG_ASK':
            score = STRONG_BEARISH
            reason = f'Order book STRONG ASK (ratio {ratio:.2f}) — heavy sell wall'
        elif imbalance == 'ASK_HEAVY':
            score = BEARISH
            reason = f'Order book ASK heavy (ratio {ratio:.2f}) — more sellers than buyers'
        else:
            score = NEUTRAL
            reason = f'Order book balanced (ratio {ratio:.2f})'

        return SignalFactor('Order Book', 'FLOW',
                            f'{imbalance} ({ratio:.2f})', score, 1.0, reason)

    def _score_fear_greed(self, fg: Dict) -> SignalFactor:
        """Signal #11: Fear & Greed Index (contrarian) — weight 0.5"""
        value = int(fg.get('value', 50) or 50)
        label = fg.get('label', 'Neutral')

        # Contrarian interpretation: extreme fear = potential buy, extreme greed = potential sell
        if value <= 20:
            score = STRONG_BULLISH
            reason = f'F&G {value} ({label}) — extreme fear, contrarian BUY signal'
        elif value <= 35:
            score = BULLISH
            reason = f'F&G {value} ({label}) — fear zone, contrarian buy opportunity'
        elif value >= 80:
            score = STRONG_BEARISH
            reason = f'F&G {value} ({label}) — extreme greed, contrarian SELL warning'
        elif value >= 65:
            score = BEARISH
            reason = f'F&G {value} ({label}) — greed zone, caution warranted'
        else:
            score = NEUTRAL
            reason = f'F&G {value} ({label}) — neutral sentiment'

        return SignalFactor('Fear & Greed', 'SENTIMENT',
                            f'{value} ({label})', score, 0.5, reason)

    def _score_btc_context(self, coin: str) -> SignalFactor:
        """Signal #12: BTC macro context (for alts) — weight 1.5"""
        if coin == 'BTC_USDT':
            # For BTC itself, use its own multi-TF trend instead
            regime = self.engine.get_market_regime(coin)
            adx = regime.get('adx', 0)
            vol = regime.get('volatility', 'MODERATE')
            score = NEUTRAL
            reason = f'BTC self-context: ADX={adx:.0f}, volatility={vol}'
            return SignalFactor('BTC Context', 'MACRO',
                                'Self (BTC)', score, 1.5, reason)

        btc = self.engine.get_btc_context()
        trend = btc.get('overall_trend', 'NEUTRAL')
        change = float(btc.get('change_24h', 0) or 0)
        support_alts = btc.get('support_alts', True)

        if trend == 'BULLISH' and change >= 1.0:
            score = STRONG_BULLISH
            reason = f'BTC {change:+.1f}% (24h), trend BULLISH — strong alt tailwind'
        elif trend == 'BULLISH':
            score = BULLISH
            reason = f'BTC {change:+.1f}% (24h), trend BULLISH — supports alts'
        elif trend == 'BEARISH' and change <= -1.0:
            score = STRONG_BEARISH
            reason = f'BTC {change:+.1f}% (24h), trend BEARISH — major alt headwind'
        elif trend == 'BEARISH':
            score = BEARISH
            reason = f'BTC {change:+.1f}% (24h), trend BEARISH — headwind for alts'
        else:
            score = NEUTRAL
            reason = f'BTC {change:+.1f}% (24h), trend NEUTRAL — no directional bias'

        return SignalFactor('BTC Context', 'MACRO',
                            f'{trend} ({change:+.1f}%)', score, 1.5, reason)

    def _score_regime_gate(self, coin: str, feat: Dict) -> SignalFactor:
        """Signal #13: Macro regime gate (weekly SMA50) — weight 1.5"""
        col = REGIME_GATE_COL.get(coin, '1w_dist_sma_50')
        dist = float(feat.get(col, 0) or 0)
        tf_label = col.split('_')[0].upper()  # '1W' or '1D'

        if dist > 5:
            score = STRONG_BULLISH
            reason = (f'{tf_label} SMA50 dist +{dist:.1f}% — deep in bull regime, '
                      'longs have macro tailwind')
        elif dist > 0:
            score = BULLISH
            reason = f'{tf_label} SMA50 dist +{dist:.1f}% — above SMA50, bull regime'
        elif dist > -5:
            score = BEARISH
            reason = (f'{tf_label} SMA50 dist {dist:.1f}% — below SMA50, bearish regime, '
                      'longs face headwind')
        else:
            score = STRONG_BEARISH
            reason = (f'{tf_label} SMA50 dist {dist:.1f}% — well below SMA50, '
                      'confirmed bear regime')

        return SignalFactor('Regime Gate', 'MACRO',
                            f'{tf_label} SMA50 dist {dist:+.1f}%', score, 1.5, reason)

    # ── Verdict calculation ───────────────────────────────────────────────────

    def _calculate_verdict(self, signals: List[SignalFactor], model_status: str) -> Dict:
        """
        Fuse 13 signal scores into a verdict.
        Confidence formula (Upgrade 2):
          score_strength = abs(weighted_score) / max_possible_score
          agreement      = max(bullish, bearish) / total
          confidence     = (score_strength * 0.6 + agreement * 0.4) * 100
          penalty        if ML disagrees with consensus
        """
        weighted_sum = sum(s.weighted_score for s in signals)
        max_score    = sum(s.weight * STRONG_BULLISH for s in signals)  # = 36.0

        bullish_count = sum(1 for s in signals if s.score > 0)
        bearish_count = sum(1 for s in signals if s.score < 0)
        neutral_count = sum(1 for s in signals if s.score == 0)
        total         = len(signals)

        # Confidence
        score_strength = abs(weighted_sum) / max_score if max_score > 0 else 0
        agreement      = max(bullish_count, bearish_count) / total if total > 0 else 0
        confidence_raw = (score_strength * 0.6 + agreement * 0.4) * 100

        # ML disagrees with consensus: 30% penalty
        ml_signal = next((s.score for s in signals if s.name == 'ML Model'), 0)
        consensus_dir = 'BULLISH' if weighted_sum > 0 else 'BEARISH'
        ml_dir        = 'BULLISH' if ml_signal > 0 else 'BEARISH' if ml_signal < 0 else 'NEUTRAL'
        if model_status == 'MARGINAL' and ml_dir != 'NEUTRAL' and ml_dir != consensus_dir:
            confidence_raw *= 0.7

        confidence = min(90, max(5, int(confidence_raw)))

        agreement_pct   = round(max(bullish_count, bearish_count) / total * 100, 1)
        agreement_label = ('HIGH' if agreement_pct >= 70
                           else 'MODERATE' if agreement_pct >= 55
                           else 'LOW')

        # Verdict thresholds (scaled to max_score=36)
        s = weighted_sum
        if model_status == 'NOT_VIABLE' and abs(s) < max_score * 0.25:
            verdict = 'AVOID'
        elif s >= max_score * 0.55:
            verdict = 'STRONG_BUY'
        elif s >= max_score * 0.28:
            verdict = 'BUY'
        elif s >= max_score * 0.11:
            verdict = 'LEAN_BUY'
        elif s <= -(max_score * 0.55):
            verdict = 'STRONG_SELL'
        elif s <= -(max_score * 0.28):
            verdict = 'SELL'
        elif s <= -(max_score * 0.11):
            verdict = 'LEAN_SELL'
        else:
            verdict = 'HOLD'

        return {
            'verdict':         verdict,
            'score':           weighted_sum,
            'max_score':       round(max_score, 1),
            'confidence':      confidence,
            'bullish_count':   bullish_count,
            'bearish_count':   bearish_count,
            'neutral_count':   neutral_count,
            'agreement_pct':   agreement_pct,
            'agreement_label': agreement_label,
        }

    # ── Trade setup ───────────────────────────────────────────────────────────

    def _calculate_trade_setup(self, price: float, verdict: str,
                                coin: str, trade_type: str, capital: float) -> Optional[Dict]:
        """Calculate entry/SL/TP and position size. Only for actionable verdicts."""
        if verdict in ('HOLD', 'AVOID') or price <= 0 or capital <= 0:
            return None

        is_long = verdict in ('STRONG_BUY', 'BUY', 'LEAN_BUY')

        # Prefer coin-specific TP/SL, fallback to trade_type
        cfg = COIN_TP_SL.get(coin, TRADE_TYPE_TP_SL.get(trade_type, {'tp_pct': 10.0, 'sl_pct': 4.0}))
        tp_pct = cfg['tp_pct']
        sl_pct = cfg['sl_pct']

        # Adjust for LEAN signals: half the target
        if verdict in ('LEAN_BUY', 'LEAN_SELL'):
            tp_pct = tp_pct * 0.6
            sl_pct = sl_pct * 0.8

        position_pct = TRADE_TYPE_POSITION_PCT.get(trade_type, 5.0)
        position_usd = capital * position_pct / 100
        max_loss_usd = position_usd * sl_pct / 100
        rr_ratio     = round(tp_pct / sl_pct, 2)

        if is_long:
            sl_price = price * (1 - sl_pct / 100)
            tp_price = price * (1 + tp_pct / 100)
        else:
            sl_price = price * (1 + sl_pct / 100)
            tp_price = price * (1 - tp_pct / 100)

        return {
            'direction':      'LONG' if is_long else 'SHORT',
            'entry_price':    round(price, 8),
            'stop_loss':      round(sl_price, 8),
            'take_profit':    round(tp_price, 8),
            'sl_pct':         sl_pct,
            'tp_pct':         tp_pct,
            'position_usd':   round(position_usd, 2),
            'position_pct':   position_pct,
            'max_loss_usd':   round(max_loss_usd, 2),
            'risk_reward':    rr_ratio,
            'max_hold_h':     cfg.get('max_hold_h', 48),
        }

    # ── AI reasoning ─────────────────────────────────────────────────────────

    async def _get_ai_reasoning(self, coin: str, price: float,
                                 verdict_data: Dict, signals: List[SignalFactor]) -> Dict:
        """Get Groq-generated reasoning. Fallback to template if unavailable."""
        if not self.groq_client:
            return self._template_reasoning(verdict_data, signals)

        bullish = [s for s in signals if s.score > 0]
        bearish = [s for s in signals if s.score < 0]

        bull_str = '; '.join(f"{s.name}: {s.reason[:60]}" for s in bullish[:4])
        bear_str = '; '.join(f"{s.name}: {s.reason[:60]}" for s in bearish[:4])

        score    = verdict_data['score']
        max_s    = verdict_data['max_score']
        verdict  = verdict_data['verdict']
        conf     = verdict_data['confidence']
        agree    = verdict_data['agreement_label']
        bull_c   = verdict_data['bullish_count']
        bear_c   = verdict_data['bearish_count']

        prompt = f"""Coin: {coin.replace('_USDT', '')} @ ${price:,.4g}
Verdict: {verdict} | Confidence: {conf}% | Agreement: {agree}
Score: {score:.1f}/{max_s} ({bull_c} bullish, {bear_c} bearish signals)
BULLISH: {bull_str or 'None'}
BEARISH: {bear_str or 'None'}

Reply with ONLY valid JSON (no markdown):
{{"summary":"2 sentences explaining the verdict with specific data","reasons":["reason1","reason2","reason3"],"risks":["risk1","risk2"],"watch_for":["trigger1","trigger2"]}}"""

        try:
            resp = self.groq_client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.3,
                max_tokens=400,
            )
            text = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith('```'):
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            print(f"VerdictEngine AI error: {e}")
            return self._template_reasoning(verdict_data, signals)

    def _template_reasoning(self, verdict_data: Dict,
                             signals: List[SignalFactor]) -> Dict:
        """Rule-based fallback when Groq is unavailable."""
        verdict  = verdict_data['verdict']
        bull_c   = verdict_data['bullish_count']
        bear_c   = verdict_data['bearish_count']
        total    = bull_c + bear_c + verdict_data['neutral_count']
        conf     = verdict_data['confidence']
        agree    = verdict_data['agreement_label']

        bullish = [s for s in signals if s.score > 0]
        bearish = [s for s in signals if s.score < 0]

        summary = (f"{verdict.replace('_', ' ').title()} verdict with {conf}% confidence "
                   f"({agree} agreement: {bull_c}/{total} signals bullish, "
                   f"{bear_c}/{total} bearish). "
                   f"{'Signals align for a long setup.' if verdict in ('BUY','STRONG_BUY','LEAN_BUY') else 'Signals align for a short/avoid setup.'}")

        reasons  = [s.reason for s in (bullish if verdict in ('BUY','STRONG_BUY','LEAN_BUY') else bearish)[:3]]
        risks    = [s.reason for s in (bearish if verdict in ('BUY','STRONG_BUY','LEAN_BUY') else bullish)[:2]]
        watch_for = [
            'BTC breaks key level — reassess all alts',
            f'Funding rate spikes above 0.05% — long squeeze risk',
        ]
        return {
            'summary':   summary,
            'reasons':   reasons or ['Insufficient signals for reasoning'],
            'risks':     risks   or ['Mixed signal environment'],
            'watch_for': watch_for,
        }

    # ── Accuracy tracking ─────────────────────────────────────────────────────

    def _store_verdict(self, coin: str, result: Dict, trade_type: str):
        """Store verdict in SQLite for accuracy tracking."""
        try:
            conn = sqlite3.connect('data/agent_memory.db')
            conn.execute("""
                INSERT INTO verdict_history
                    (coin, timestamp, verdict, confidence, consensus_score,
                     price_at_verdict, trade_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                coin,
                result['timestamp'],
                result['verdict'],
                result['confidence'],
                result['consensus_score'],
                result['price'],
                trade_type,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"VerdictEngine store error: {e}")

    def _check_accuracy_async(self, coin: str):
        """
        Check old verdicts (24h, 48h, 7d) and update was_correct.
        Runs inline — quick query, only updates rows missing price_after_*.
        """
        try:
            now    = datetime.now()
            price, _ = self.engine.get_live_price(coin)
            if price <= 0:
                return

            conn = sqlite3.connect('data/agent_memory.db')
            rows = conn.execute("""
                SELECT id, timestamp, verdict, price_at_verdict,
                       price_after_24h, price_after_48h, price_after_7d
                FROM verdict_history
                WHERE coin = ? AND was_correct IS NULL
                ORDER BY id DESC LIMIT 20
            """, (coin,)).fetchall()

            for row in rows:
                rid, ts_str, verdict, entry, p24, p48, p7d = row
                try:
                    ts = datetime.fromisoformat(ts_str)
                except Exception:
                    continue

                hours_elapsed = (now - ts).total_seconds() / 3600
                updates = {}

                if hours_elapsed >= 24 and p24 is None:
                    updates['price_after_24h'] = price
                if hours_elapsed >= 48 and p48 is None:
                    updates['price_after_48h'] = price
                if hours_elapsed >= 168 and p7d is None:
                    updates['price_after_7d'] = price

                if updates and entry and entry > 0:
                    # Determine correctness at 48h (primary check)
                    check_price = updates.get('price_after_48h') or updates.get('price_after_24h')
                    if check_price:
                        chg_pct = (check_price - entry) / entry * 100
                        is_bull = verdict in ('STRONG_BUY', 'BUY', 'LEAN_BUY')
                        is_bear = verdict in ('STRONG_SELL', 'SELL', 'LEAN_SELL')
                        correct = ((is_bull and chg_pct > 0.5) or
                                   (is_bear and chg_pct < -0.5) or
                                   (not is_bull and not is_bear))
                        updates['was_correct'] = 1 if correct else 0
                        updates['checked_at']  = now.isoformat()

                if updates:
                    set_clause = ', '.join(f"{k}=?" for k in updates)
                    conn.execute(
                        f"UPDATE verdict_history SET {set_clause} WHERE id=?",
                        list(updates.values()) + [rid]
                    )

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"VerdictEngine accuracy check error: {e}")

    def get_accuracy_stats(self, coin: str, days: int = 30) -> Dict:
        """Return accuracy stats for a coin over last N days."""
        try:
            conn   = sqlite3.connect('data/agent_memory.db')
            cutoff = datetime.now().strftime('%Y-%m-%dT00:00:00')
            rows   = conn.execute("""
                SELECT verdict, was_correct, confidence
                FROM verdict_history
                WHERE coin = ? AND timestamp >= ? AND was_correct IS NOT NULL
            """, (coin, cutoff)).fetchall()
            conn.close()

            if not rows:
                return {'total': 0, 'correct': 0, 'accuracy_pct': 0, 'by_verdict': {}}

            total   = len(rows)
            correct = sum(1 for r in rows if r[1] == 1)
            by_v: Dict[str, Dict] = {}
            for v, c, conf in rows:
                if v not in by_v:
                    by_v[v] = {'total': 0, 'correct': 0}
                by_v[v]['total'] += 1
                if c == 1:
                    by_v[v]['correct'] += 1

            return {
                'total':        total,
                'correct':      correct,
                'accuracy_pct': round(correct / total * 100, 1),
                'by_verdict':   by_v,
                'days':         days,
            }
        except Exception as e:
            print(f"VerdictEngine accuracy stats error: {e}")
            return {'total': 0, 'correct': 0, 'accuracy_pct': 0, 'by_verdict': {}}
