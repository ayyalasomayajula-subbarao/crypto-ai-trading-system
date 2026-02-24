import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import PriceChart from './PriceChart';
import { useAuth } from '../context/AuthContext';
import { db } from '../lib/supabase';
import './Coinpage.css';

const API_BASE = process.env.REACT_APP_API_URL || window.location.origin;
const WS_URL = process.env.REACT_APP_WS_URL || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/prices`;

// ============================================================
// TYPES
// ============================================================

interface PriceData {
  price: number;
  change_24h: number;
  direction: 'up' | 'down' | 'same';
  timestamp: string;
}

interface ActiveScenario {
  type: string;
  title: string;
  message: string;
  icon: string;
  severity: string;
  effect: string;
  details?: string;
}

interface TradeTypeRequirement {
  name: string;
  required: string;
  current: string;
  met: boolean;
}

interface TradeTypeInfo {
  name: string;
  duration_hours: number;
  duration_display: string;
  description: string;
  risk_level: string;
  base_threshold: number;
  min_adx: number;
  min_expectancy: number;
  max_loss_prob: number;
}

interface ExperienceInfo {
  name: string;
  description: string;
  threshold_boost: number;
  position_mult: number;
}

interface AIPositive {
  title: string;
  detail: string;
}

interface AIRisk {
  title: string;
  detail: string;
  severity: string;
}

interface AIMarketPulse {
  trend_summary: string;
  volume_verdict: string;
  sentiment_verdict: string;
  macro_context: string;
}

interface AIAnalysis {
  ai_verdict?: string;
  confidence_score?: number;
  tldr?: string;
  summary?: string; // backward compat
  key_insights?: string[];
  positives?: AIPositive[];
  risks?: AIRisk[];
  market_pulse?: AIMarketPulse;
  risk_assessment?: string; // backward compat
  entry_strategy?: string;
  what_to_watch?: string[];
  sentiment_read?: string;
  conviction_reason?: string;
  ai_provider?: string;
  ai_cost?: string;
  error?: string;
}

interface AIAnalysisResponse {
  coin: string;
  timestamp: string;
  ai_analysis: AIAnalysis;
  technical_summary: {
    verdict: string;
    win_probability: number;
    loss_probability: number;
    expectancy: number;
    price: number;
  };
}

interface VolumeProfileLevel {
  price: number;
  price_low: number;
  price_high: number;
  volume: number;
  volume_pct: number;
  volume_relative: number;
  is_poc: boolean;
}

interface VolumeProfile {
  poc: {
    price: number;
    volume: number;
    volume_pct: number;
  };
  value_area_high: number;
  value_area_low: number;
  current_price: number;
  price_vs_poc: string;
  levels: VolumeProfileLevel[];
  hvn: { price: number; volume_pct: number }[];
  lvn: { price: number; volume_pct: number }[];
  data_source?: string;
  analysis: string;
}

interface VolumeAnalysis {
  obv: {
    current: number;
    trend: string;
    divergence: string;
  };
  mfi: {
    value: number;
    zone: string;
    interpretation: string;
  };
  buy_sell_delta: {
    buy_volume: number;
    sell_volume: number;
    delta_pct: number;
    pressure: string;
    strength: string;
  };
  volume_spikes: {
    current_ratio: number;
    is_spike: boolean;
    spike_count_24h: number;
  };
  force_index: {
    value: number;
    trend: string;
  };
  volume_profile?: VolumeProfile;
  overall_signal: string;
  summary: string;
}

interface MarketSentiment {
  fear_greed: {
    value: number;
    label: string;
    trend?: string;
    error?: string;
  };
  funding_rate: {
    rate: number;
    rate_pct: string;
    sentiment: string;
    interpretation: string;
    error?: string;
  };
  open_interest: {
    value: number;
    formatted: string;
    error?: string;
  };
  overall_sentiment: string;
  timestamp: string;
}

interface DerivativesIntelligence {
  long_short_ratio: {
    top_traders: { long_pct: number; short_pct: number; ratio: number; trend: string };
    global: { long_pct: number; short_pct: number; ratio: number; trend: string };
    signal: string;
  };
  taker_volume: {
    buy_vol: number;
    sell_vol: number;
    ratio: number;
    pressure: string;
    trend: string;
  };
  order_book: {
    total_bid_usd: number;
    total_ask_usd: number;
    bid_ask_ratio: number;
    imbalance: string;
    strongest_bid: { price: number; size_usd: number };
    strongest_ask: { price: number; size_usd: number };
    support_level: number;
    resistance_level: number;
  };
  liquidations: {
    long_liq_levels: { leverage: string; price: number; distance_pct: number }[];
    short_liq_levels: { leverage: string; price: number; distance_pct: number }[];
    recent_signal: string;
    oi_change_pct: number;
    nearest_long_liq: { leverage: string; price: number; distance_pct: number } | null;
    nearest_short_liq: { leverage: string; price: number; distance_pct: number } | null;
  };
  overall_signal: string;
  timestamp: string;
}

interface WhaleActivity {
  recent_large_txs: {
    hash?: string;
    value: number;
    value_display: string;
    time: string;
    type?: string;
    block?: number;
  }[];
  whale_signal: string;
  large_trade_count: number;
  avg_large_trade_size: number;
  alert: string | null;
  source: string;
}

interface AnalysisResult {
  coin: string;
  timestamp: string;
  price: number;
  price_source: string;
  capital: number;
  trade_type: string;
  experience_level: string;
  trade_reason?: string;
  verdict: string;
  confidence: string;
  model_ran: boolean;
  blocked_by?: string;
  block_reason?: string;
  win_probability: number | null;
  loss_probability: number | null;
  sideways_probability?: number | null;
  win_threshold_used: number | null;
  expectancy: number | null;
  expectancy_status: string | null;
  readiness: number | null;
  readiness_status: string | null;
  risk_adjusted_ev: number | null;
  reasoning: string[];
  warnings: string[];
  active_scenarios: ActiveScenario[];
  scenario_count: number;
  trade_type_info?: TradeTypeInfo;
  experience_info?: ExperienceInfo;
  trade_type_requirements?: TradeTypeRequirement[];
  forecast: {
    direction: string;
    current_price: number;
    bull_target: number;
    bear_target: number;
    probabilities: {
      up: number;
      sideways: number;
      down: number;
    };
  };
  risk: {
    action: string;
    position_size_usd?: number;
    position_size_pct?: number;
    entry_price?: number;
    stop_loss_price?: number;
    stop_loss_pct?: number;
    take_profit_price?: number;
    take_profit_pct?: number;
    max_loss_usd?: number;
    max_hold_hours?: number;
  };
  suggested_action: {
    action: string;
    message: string;
    next_check: string | null;
    conditions?: string[];
  };
  market_context: {
    btc: {
      trend_1h: string;
      trend_4h: string;
      trend_1d: string;
      overall_trend: string;
      support_alts: boolean;
      price: number;
      change_24h: number;
    };
    regime: {
      regime: string;
      adx: number;
      volatility: string;
      volatility_pct?: number;
      recommendation?: string;
    };
  };
  volume_analysis?: VolumeAnalysis;
}

// ============================================================
// COIN CONFIGURATION
// ============================================================

const COIN_CONFIG: Record<string, { icon: string; name: string; color: string; description: string }> = {
  'BTC_USDT': {
    icon: '₿',
    name: 'Bitcoin',
    color: '#f7931a',
    description: 'The original cryptocurrency. Lower volatility, market leader.'
  },
  'ETH_USDT': {
    icon: 'Ξ',
    name: 'Ethereum',
    color: '#627eea',
    description: 'Smart contract platform. Moderate volatility, follows BTC.'
  },
  'SOL_USDT': {
    icon: '◎',
    name: 'Solana',
    color: '#00ffa3',
    description: 'High-performance blockchain. Higher volatility than BTC/ETH.'
  },
  'PEPE_USDT': {
    icon: '🐸',
    name: 'Pepe',
    color: '#4a9c2d',
    description: 'Meme coin. Extreme volatility, highest risk/reward.'
  }
};

const TRADE_TYPE_INFO: Record<string, { name: string; duration: string; description: string; riskLevel: string }> = {
  'SCALP': {
    name: 'Scalp',
    duration: 'Minutes to hours',
    description: 'Quick in-and-out trades capturing small price movements. Requires high win rate and strict discipline.',
    riskLevel: 'Very High'
  },
  'SHORT_TERM': {
    name: 'Short Term',
    duration: '1-2 days',
    description: 'Capturing intraday swings. Balance between frequency and quality of setups.',
    riskLevel: 'High'
  },
  'SWING': {
    name: 'Swing',
    duration: '2-7 days',
    description: 'Riding medium-term trends. Most balanced approach for beginners and intermediates.',
    riskLevel: 'Medium'
  },
  'INVESTMENT': {
    name: 'Investment',
    duration: 'Weeks to months',
    description: 'Long-term position building. Lower win rate acceptable with larger targets.',
    riskLevel: 'Lower'
  }
};

// ============================================================
// TOOLTIP COMPONENT
// ============================================================

const Tooltip: React.FC<{ text: string; children: React.ReactNode }> = ({ text, children }) => {
  const [show, setShow] = useState(false);
  
  return (
    <div 
      className="tooltip-container"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      <span className="tooltip-icon">ⓘ</span>
      {show && <div className="tooltip-content">{text}</div>}
    </div>
  );
};

// ============================================================
// MAIN COMPONENT
// ============================================================

const CoinPage: React.FC = () => {
  const { coinId } = useParams<{ coinId: string }>();
  const { user } = useAuth();
  const coin = coinId?.toUpperCase() || 'BTC_USDT';
  const coinConfig = COIN_CONFIG[coin] || COIN_CONFIG['BTC_USDT'];

  // State
  const [price, setPrice] = useState<PriceData | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [lastAnalysis, setLastAnalysis] = useState<Date | null>(null);

  // AI Analysis State
  const [aiAnalysis, setAiAnalysis] = useState<AIAnalysis | null>(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiAvailable, setAiAvailable] = useState(false);

  // Market Sentiment State
  const [marketSentiment, setMarketSentiment] = useState<MarketSentiment | null>(null);
  const [sentimentLoading, setSentimentLoading] = useState(false);

  // Derivatives & Whale State
  const [derivativesData, setDerivativesData] = useState<DerivativesIntelligence | null>(null);
  const [derivativesLoading, setDerivativesLoading] = useState(false);
  const [whaleData, setWhaleData] = useState<WhaleActivity | null>(null);
  const [whaleLoading, setWhaleLoading] = useState(false);

  // User Inputs
  const [capital, setCapital] = useState<number | ''>('');
  const [tradeType, setTradeType] = useState('SWING');
  const [experience, setExperience] = useState('INTERMEDIATE');
  const [reason, setReason] = useState('');
  const [recentLosses, setRecentLosses] = useState(0);
  const [tradesToday, setTradesToday] = useState(0);

  // WebSocket
  const wsRef = useRef<WebSocket | null>(null);

  // ============================================================
  // WEBSOCKET CONNECTION
  // ============================================================
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        setWsConnected(true);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'initial' && data.prices[coin]) {
          setPrice(data.prices[coin]);
        } else if (data.type === 'price_update' && data.coin === coin) {
          setPrice(data.data);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        setTimeout(connectWebSocket, 3000);
      };

      wsRef.current = ws;
    };

    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [coin]);

  // ============================================================
  // FETCH ANALYSIS
  // ============================================================
  const fetchAnalysis = useCallback(async () => {
    if (!capital || capital <= 0) {
      alert('Please enter your capital amount first');
      return;
    }
    
    setLoading(true);
    try {
      const params = new URLSearchParams({
        capital: capital.toString(),
        trade_type: tradeType,
        experience: experience,
        recent_losses: recentLosses.toString(),
        trades_today: tradesToday.toString()
      });

      if (reason) params.append('reason', reason);

      const response = await axios.get<AnalysisResult>(
        `${API_BASE}/analyze/${coin}?${params}`
      );
      setAnalysis(response.data);
      setLastAnalysis(new Date());

      // Auto-save signal to Supabase for history tracking
      if (user && response.data) {
        const d = response.data;
        db.saveSignal({
          user_id: user.id,
          coin: d.coin,
          trade_type: d.trade_type,
          verdict: d.verdict,
          confidence: d.confidence,
          win_probability: d.win_probability,
          loss_probability: d.loss_probability,
          sideways_probability: d.sideways_probability || null,
          expectancy: d.expectancy,
          price_at_signal: d.price,
          market_regime: d.market_context?.regime?.regime || null,
          reasoning: { reasons: d.reasoning, warnings: d.warnings }
        }).catch(err => console.error('Signal save error:', err));
      }
    } catch (error) {
      console.error('Error fetching analysis:', error);
    } finally {
      setLoading(false);
    }
  }, [coin, capital, tradeType, experience, reason, recentLosses, tradesToday, user]);

  // Fetch AI Analysis
  const fetchAIAnalysis = useCallback(async () => {
    if (!capital || capital <= 0) {
      alert('Please enter your capital and run the standard analysis first');
      return;
    }

    setAiLoading(true);
    try {
      const params = new URLSearchParams({
        capital: capital.toString(),
        trade_type: tradeType,
        experience: experience,
        recent_losses: recentLosses.toString(),
        trades_today: tradesToday.toString()
      });

      if (reason) params.append('reason', reason);

      const response = await axios.get<AIAnalysisResponse>(
        `${API_BASE}/ai-analysis/${coin}?${params}`
      );
      setAiAnalysis(response.data.ai_analysis);
    } catch (error: any) {
      console.error('Error fetching AI analysis:', error);
      const message = error?.response?.data?.detail || 'AI analysis failed. Check that API keys are configured.';
      setAiAnalysis({
        error: message,
        ai_provider: 'None',
        ai_cost: 'N/A'
      });
    } finally {
      setAiLoading(false);
    }
  }, [coin, capital, tradeType, experience, reason, recentLosses, tradesToday]);

  // Fetch Market Sentiment
  const fetchMarketSentiment = useCallback(async () => {
    setSentimentLoading(true);
    try {
      const response = await axios.get<MarketSentiment>(
        `${API_BASE}/market-sentiment/${coin}`
      );
      setMarketSentiment(response.data);
    } catch (error) {
      console.error('Error fetching market sentiment:', error);
    } finally {
      setSentimentLoading(false);
    }
  }, [coin]);

  const fetchDerivatives = useCallback(async () => {
    setDerivativesLoading(true);
    try {
      const response = await axios.get<DerivativesIntelligence>(
        `${API_BASE}/derivatives/${coin}`
      );
      setDerivativesData(response.data);
    } catch (error) {
      console.error('Error fetching derivatives:', error);
    } finally {
      setDerivativesLoading(false);
    }
  }, [coin]);

  const fetchWhaleActivity = useCallback(async () => {
    setWhaleLoading(true);
    try {
      const response = await axios.get<WhaleActivity>(
        `${API_BASE}/whales/${coin}`
      );
      setWhaleData(response.data);
    } catch (error) {
      console.error('Error fetching whale activity:', error);
    } finally {
      setWhaleLoading(false);
    }
  }, [coin]);

  // Fetch sentiment, derivatives, and whale data when analysis completes
  useEffect(() => {
    if (analysis) {
      fetchMarketSentiment();
      fetchDerivatives();
      fetchWhaleActivity();
    }
  }, [analysis, fetchMarketSentiment, fetchDerivatives, fetchWhaleActivity]);

  // Check AI availability on mount
  useEffect(() => {
    axios.get(`${API_BASE}/ai-status`)
      .then(res => setAiAvailable(res.data.any_available))
      .catch(() => setAiAvailable(false));
  }, []);

  // Initial fetch only if capital is set
  useEffect(() => {
    if (capital && capital > 0) {
      fetchAnalysis();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [coin]); // Only on coin change, not on every input change

  // ============================================================
  // HELPERS
  // ============================================================
  const formatPrice = (p: number): string => {
    if (p < 0.0001) return `$${p.toFixed(8)}`;
    if (p < 0.01) return `$${p.toFixed(6)}`;
    if (p < 1) return `$${p.toFixed(4)}`;
    if (p < 1000) return `$${p.toFixed(2)}`;
    return `$${p.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  };

  const getVerdictColor = (verdict: string): string => {
    switch (verdict) {
      case 'BUY': return '#10b981';
      case 'WAIT': return '#f59e0b';
      case 'AVOID': return '#ef4444';
      case 'BLOCKED': return '#8b5cf6';
      default: return '#6b7280';
    }
  };

  const getSeverityColor = (severity: string): string => {
    switch (severity) {
      case 'CRITICAL': return '#ef4444';
      case 'HIGH': return '#f59e0b';
      case 'MEDIUM': return '#3b82f6';
      default: return '#6b7280';
    }
  };

  const getSignalColor = (signal: string): string => {
    switch (signal) {
      case 'BULLISH': case 'ACCUMULATION': return '#10b981';
      case 'BEARISH': case 'DISTRIBUTION': return '#ef4444';
      case 'OVERBOUGHT': return '#f59e0b';
      case 'OVERSOLD': return '#3b82f6';
      default: return '#6b7280';
    }
  };

  const getFearGreedColor = (value: number): string => {
    if (value <= 25) return '#ef4444';
    if (value <= 45) return '#f59e0b';
    if (value <= 55) return '#6b7280';
    if (value <= 75) return '#22c55e';
    return '#10b981';
  };

  const tradeTypeInfo = TRADE_TYPE_INFO[tradeType];

  // ============================================================
  // RENDER
  // ============================================================
  return (
    <div className="coin-page">
      {/* Header */}
      <header className="coin-header">
        <div className="coin-title">
          <span className="coin-icon" style={{ color: coinConfig.color }}>{coinConfig.icon}</span>
          <div>
            <h1>{coin.replace('_', '/')}</h1>
            <span className="coin-name">{coinConfig.name}</span>
          </div>
        </div>
        <div className="header-actions">
          <span className={`ws-badge ${wsConnected ? 'connected' : ''}`}>
            {wsConnected ? '🟢 Live' : '🔴 Offline'}
          </span>
          <button className="analyze-btn" onClick={fetchAnalysis} disabled={loading}>
            {loading ? '⏳ Analyzing...' : '🔄 Analyze'}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="coin-content">
        {/* LEFT: Price & User Inputs */}
        <aside className="coin-sidebar">
          {/* Live Price Card */}
          <div className={`price-card ${price?.direction || ''}`}>
            <div className="price-label">Live Price</div>
            <div className="price-value">
              {price ? formatPrice(price.price) : '—'}
            </div>
            {price && (
              <div className={`price-change ${price.change_24h >= 0 ? 'up' : 'down'}`}>
                {price.change_24h >= 0 ? '▲' : '▼'} {Math.abs(price.change_24h).toFixed(2)}% (24h)
              </div>
            )}
            <div className="price-source">Source: Binance WebSocket</div>
          </div>

          {/* Coin Description */}
          <div className="info-card">
            <h3>About {coinConfig.name}</h3>
            <p>{coinConfig.description}</p>
          </div>

          {/* User Inputs */}
          <div className="inputs-card">
            <h3>⚙️ Your Trade Settings</h3>
            
            <div className="input-group">
              <Tooltip text="Enter your total trading capital. Position sizing will be calculated based on this amount and your trade type.">
                <label>Your Capital ($) *</label>
              </Tooltip>
              <input
                type="number"
                value={capital}
                onChange={(e) => setCapital(e.target.value ? Number(e.target.value) : '')}
                placeholder="Enter your capital (e.g., 5000)"
                min={100}
                step={100}
                required
              />
              {!capital && (
                <span className="input-hint">Required - Enter your trading capital</span>
              )}
            </div>

            <div className="input-group">
              <Tooltip text="Different trade types have different WIN probability thresholds and position sizing rules.">
                <label>Trade Type</label>
              </Tooltip>
              <select value={tradeType} onChange={(e) => setTradeType(e.target.value)}>
                <option value="SCALP">🎯 Scalp (minutes)</option>
                <option value="SHORT_TERM">⚡ Short Term (1-2 days)</option>
                <option value="SWING">📊 Swing (2-7 days)</option>
                <option value="INVESTMENT">💎 Investment (weeks+)</option>
              </select>
            </div>

            {/* Trade Type Info Box */}
            <div className="trade-type-info">
              <div className="tti-header">
                <span>{tradeTypeInfo.name}</span>
                <span className={`risk-badge ${tradeTypeInfo.riskLevel.toLowerCase().replace(' ', '-')}`}>
                  {tradeTypeInfo.riskLevel} Risk
                </span>
              </div>
              <div className="tti-duration">⏱️ {tradeTypeInfo.duration}</div>
              <div className="tti-desc">{tradeTypeInfo.description}</div>
            </div>

            <div className="input-group">
              <Tooltip text="Your experience level affects required WIN thresholds. Beginners need higher probability setups.">
                <label>Experience Level</label>
              </Tooltip>
              <select value={experience} onChange={(e) => setExperience(e.target.value)}>
                <option value="BEGINNER">🌱 Beginner</option>
                <option value="INTERMEDIATE">📈 Intermediate</option>
                <option value="ADVANCED">🎓 Advanced</option>
              </select>
            </div>

            <div className="input-group">
              <Tooltip text="Be honest! If you're feeling FOMO, the system will raise thresholds to protect you from emotional trades.">
                <label>Why This Trade?</label>
              </Tooltip>
              <select value={reason} onChange={(e) => setReason(e.target.value)}>
                <option value="">📋 Strategy / Analysis</option>
                <option value="FOMO">😰 FOMO (Fear of Missing Out)</option>
                <option value="NEWS">📰 News / Event</option>
                <option value="TIP">💬 Someone's Tip</option>
                <option value="DIP_BUY">📉 Buying the Dip</option>
              </select>
            </div>

            <div className="input-row">
              <div className="input-group">
                <Tooltip text="Consecutive losses trigger protective measures. Be honest to avoid further losses.">
                  <label>Recent Losses</label>
                </Tooltip>
                <input
                  type="number"
                  value={recentLosses}
                  onChange={(e) => setRecentLosses(Number(e.target.value))}
                  min={0}
                  max={10}
                />
              </div>
              <div className="input-group">
                <Tooltip text="Daily trade limit protects against overtrading. Quality over quantity.">
                  <label>Trades Today</label>
                </Tooltip>
                <input
                  type="number"
                  value={tradesToday}
                  onChange={(e) => setTradesToday(Number(e.target.value))}
                  min={0}
                  max={20}
                />
              </div>
            </div>

            {/* Input Warnings */}
            {reason === 'FOMO' && (
              <div className="input-alert danger">
                🚨 FOMO detected! Thresholds will be raised to protect you.
              </div>
            )}
            {recentLosses >= 2 && (
              <div className="input-alert warning">
                ⚠️ {recentLosses} recent losses. Consider taking a break.
              </div>
            )}
            {tradesToday >= 5 && (
              <div className="input-alert warning">
                ⚠️ {tradesToday} trades today. Approaching daily limit.
              </div>
            )}

            <button 
              className="analyze-full-btn" 
              onClick={fetchAnalysis} 
              disabled={loading || !capital || capital <= 0}
            >
              {loading ? '⏳ Analyzing...' : !capital ? '💰 Enter Capital First' : '🔍 Run Full Analysis'}
            </button>
            
            {lastAnalysis && (
              <div className="last-analysis">
                Last analysis: {lastAnalysis.toLocaleTimeString()}
              </div>
            )}
          </div>
        </aside>

        {/* RIGHT: Analysis Results */}
        <main className="coin-main">
          {/* Price Chart - Always visible */}
          <PriceChart coin={coin} coinColor={coinConfig.color} />

          {loading && !analysis ? (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>Running personalized analysis...</p>
            </div>
          ) : analysis ? (
            <>
              {/* Verdict Section */}
              <section className="verdict-section">
                <div 
                  className="verdict-card"
                  style={{ 
                    borderColor: getVerdictColor(analysis.verdict),
                    backgroundColor: `${getVerdictColor(analysis.verdict)}15`
                  }}
                >
                  <div className="verdict-main">
                    <Tooltip text="The system's overall recommendation based on ML model predictions, market conditions, risk parameters, and trade type requirements. BUY = conditions met, WAIT = almost there, AVOID = unfavorable, BLOCKED = safety limit triggered.">
                      <span className="verdict-label">Recommendation</span>
                    </Tooltip>
                    <span
                      className="verdict-value"
                      style={{ color: getVerdictColor(analysis.verdict) }}
                    >
                      {analysis.verdict}
                    </span>
                    <Tooltip text="How confident the system is in its verdict. HIGH = most indicators align, MEDIUM = mixed signals, LOW = conflicting data.">
                      <span className="confidence-badge">{analysis.confidence} confidence</span>
                    </Tooltip>
                  </div>
                  
                  {analysis.verdict === 'BLOCKED' && analysis.block_reason && (
                    <div className="block-reason">
                      <span className="block-icon">🚫</span>
                      <span>{analysis.block_reason}</span>
                    </div>
                  )}
                </div>
              </section>

              {/* Probabilities Section */}
              {analysis.model_ran && (
                <section className="metrics-section">
                  <h2>📊 Probability Analysis</h2>
                  
                  <div className="prob-cards">
                    {/* WIN Probability */}
                    <div className="prob-card win">
                      <Tooltip text="The probability that this trade will hit your take-profit target based on historical patterns and current market conditions.">
                        <div className="prob-header">
                          <span className="prob-icon">📈</span>
                          <span className="prob-title">WIN Probability</span>
                        </div>
                      </Tooltip>
                      <div className="prob-value">{analysis.win_probability}%</div>
                      <div className="prob-bar">
                        <div className="prob-fill" style={{ width: `${analysis.win_probability}%` }}></div>
                      </div>
                      <div className="prob-threshold">
                        Threshold: {analysis.win_threshold_used}%
                        {(analysis.win_probability || 0) >= (analysis.win_threshold_used || 0) 
                          ? ' ✅ Met' 
                          : ' ❌ Not met'}
                      </div>
                    </div>

                    {/* LOSS Probability */}
                    <div className="prob-card loss">
                      <Tooltip text="The probability that this trade will hit your stop-loss. Lower is better. Above 45% is considered high risk.">
                        <div className="prob-header">
                          <span className="prob-icon">📉</span>
                          <span className="prob-title">LOSS Probability</span>
                        </div>
                      </Tooltip>
                      <div className="prob-value">{analysis.loss_probability}%</div>
                      <div className="prob-bar loss">
                        <div className="prob-fill" style={{ width: `${analysis.loss_probability}%` }}></div>
                      </div>
                      <div className="prob-note">
                        {(analysis.loss_probability || 0) > 45 
                          ? '⚠️ High loss probability' 
                          : '✅ Acceptable risk'}
                      </div>
                    </div>
                  </div>

                  {/* Expectancy & Readiness */}
                  <div className="metric-cards">
                    <div className={`metric-card ${(analysis.expectancy || 0) >= 0 ? 'positive' : 'negative'}`}>
                      <Tooltip text="Expectancy = WIN% - LOSS%. Positive means you're more likely to win than lose. This is the primary indicator of trade quality.">
                        <div className="metric-header">
                          <span>Expectancy</span>
                          <span className="metric-formula">WIN% − LOSS%</span>
                        </div>
                      </Tooltip>
                      <div className="metric-value">
                        {(analysis.expectancy || 0) >= 0 ? '+' : ''}{analysis.expectancy}%
                      </div>
                      <div className="metric-status">{analysis.expectancy_status}</div>
                    </div>

                    <div className={`metric-card ${(analysis.readiness || 0) >= 0 ? 'positive' : 'negative'}`}>
                      <Tooltip text="Readiness = WIN% - Required Threshold. Positive means you've met the entry requirements for your trade type and experience level.">
                        <div className="metric-header">
                          <span>Readiness</span>
                          <span className="metric-formula">WIN% − Threshold</span>
                        </div>
                      </Tooltip>
                      <div className="metric-value">
                        {(analysis.readiness || 0) >= 0 ? '+' : ''}{analysis.readiness}%
                      </div>
                      <div className="metric-status">{analysis.readiness_status}</div>
                    </div>
                  </div>
                </section>
              )}

              {/* Market Regime Section */}
              <section className="regime-section">
                <h2>📈 Market Regime</h2>
                
                <div className="regime-cards">
                  <div className="regime-card">
                    <Tooltip text="ADX (Average Directional Index) measures trend strength. Below 20 = weak/no trend. 20-40 = developing trend. Above 40 = strong trend.">
                      <div className="regime-header">
                        <span>ADX (Trend Strength)</span>
                      </div>
                    </Tooltip>
                    <div className="adx-display">
                      <div className="adx-value">{analysis.market_context.regime.adx}</div>
                      <div className="adx-meter">
                        <div className="adx-fill" style={{ width: `${Math.min(analysis.market_context.regime.adx, 60) / 60 * 100}%` }}></div>
                        <div className="adx-zones">
                          <span>Weak</span>
                          <span>Moderate</span>
                          <span>Strong</span>
                        </div>
                      </div>
                    </div>
                    <div className="adx-interpretation">
                      {analysis.market_context.regime.adx < 20 && '⚠️ Weak trend - avoid trend-following strategies'}
                      {analysis.market_context.regime.adx >= 20 && analysis.market_context.regime.adx < 40 && '📊 Moderate trend - proceed with caution'}
                      {analysis.market_context.regime.adx >= 40 && '✅ Strong trend - good for trend-following'}
                    </div>
                  </div>

                  <div className="regime-card">
                    <Tooltip text="The current market regime classification. STRONG_TREND = clear directional move, WEAK_TREND = mild direction, CHOPPY = sideways/ranging with no clear direction.">
                      <div className="regime-header">
                        <span>Market State</span>
                      </div>
                    </Tooltip>
                    <div className={`regime-badge ${analysis.market_context.regime.regime.toLowerCase()}`}>
                      {analysis.market_context.regime.regime}
                    </div>
                    <div className="volatility-info">
                      <Tooltip text="Price volatility measured as standard deviation of recent closes. LOW = stable, MODERATE = normal swings, HIGH = large moves, EXTREME = very risky conditions.">
                        <span>Volatility: <strong>{analysis.market_context.regime.volatility}</strong></span>
                      </Tooltip>
                      {analysis.market_context.regime.volatility_pct && (
                        <span> ({analysis.market_context.regime.volatility_pct}%)</span>
                      )}
                    </div>
                    {analysis.market_context.regime.recommendation && (
                      <div className="regime-rec">{analysis.market_context.regime.recommendation}</div>
                    )}
                  </div>
                </div>
              </section>

              {/* Volume Analysis Section */}
              {analysis.volume_analysis && (
                <section className="volume-section">
                  <h2>📊 Volume Analysis</h2>
                  <div className={`volume-signal-banner ${analysis.volume_analysis.overall_signal.toLowerCase()}`}>
                    <Tooltip text="Composite signal from OBV, MFI, Buy/Sell Delta, Volume Spikes, and Force Index. Shows whether volume confirms or contradicts the current price trend.">
                      <span className="volume-signal-label">Overall Volume Signal</span>
                    </Tooltip>
                    <span className="volume-signal-value" style={{ color: getSignalColor(analysis.volume_analysis.overall_signal) }}>
                      {analysis.volume_analysis.overall_signal}
                    </span>
                    <span className="volume-signal-summary">{analysis.volume_analysis.summary}</span>
                  </div>

                  {/* Volume Profile */}
                  {analysis.volume_analysis.volume_profile && analysis.volume_analysis.volume_profile.levels && analysis.volume_analysis.volume_profile.levels.length > 0 && (
                    <div className="volume-profile-section">
                      <div className="vp-header-row">
                        <h3>Volume Profile (7d)
                          {analysis.volume_analysis!.volume_profile!.data_source === 'LIVE' && (
                            <span className="vp-live-badge">LIVE DATA</span>
                          )}
                        </h3>
                        <div className="vp-legend">
                          <span className="vp-legend-item poc-legend">POC (Fair Value)</span>
                          <span className="vp-legend-item va-legend">Value Area (70%)</span>
                          <span className="vp-legend-item current-legend">Current Price</span>
                        </div>
                      </div>

                      <div className="vp-stats-row">
                        <div className="vp-stat">
                          <Tooltip text="Point of Control — the price level with the highest traded volume. Acts as a magnet for price and represents 'fair value' where most trading occurred.">
                            <span className="vp-stat-label">POC</span>
                          </Tooltip>
                          <span className="vp-stat-value">{formatPrice(analysis.volume_analysis.volume_profile.poc.price)}</span>
                          <span className="vp-stat-sub">{analysis.volume_analysis.volume_profile.poc.volume_pct}% vol</span>
                        </div>
                        <div className="vp-stat">
                          <Tooltip text="The upper boundary of the Value Area — the range where 70% of all volume was traded. Acts as resistance if price is below.">
                            <span className="vp-stat-label">Value Area High</span>
                          </Tooltip>
                          <span className="vp-stat-value">{formatPrice(analysis.volume_analysis.volume_profile.value_area_high)}</span>
                        </div>
                        <div className="vp-stat">
                          <Tooltip text="The lower boundary of the Value Area. Acts as support if price is above. A break below may trigger accelerated selling.">
                            <span className="vp-stat-label">Value Area Low</span>
                          </Tooltip>
                          <span className="vp-stat-value">{formatPrice(analysis.volume_analysis.volume_profile.value_area_low)}</span>
                        </div>
                        <div className="vp-stat">
                          <Tooltip text="Where the current price is relative to the Point of Control. ABOVE_POC = potentially overvalued, BELOW_POC = potentially undervalued, AT_POC = at fair value.">
                            <span className="vp-stat-label">Price vs POC</span>
                          </Tooltip>
                          <span className={`vp-stat-value vp-pos-${analysis.volume_analysis.volume_profile.price_vs_poc.toLowerCase().replace('_', '-')}`}>
                            {analysis.volume_analysis.volume_profile.price_vs_poc.replace('_', ' ')}
                          </span>
                        </div>
                      </div>

                      <div className="vp-chart">
                        {analysis.volume_analysis!.volume_profile!.levels.map((level, i) => {
                          const vp = analysis.volume_analysis!.volume_profile!;
                          const isInValueArea = level.price >= vp.value_area_low && level.price <= vp.value_area_high;
                          const isCurrentPrice = Math.abs(level.price - vp.current_price) / vp.current_price < 0.015;
                          const isHvn = vp.hvn.some(h => Math.abs(h.price - level.price) / level.price < 0.005);
                          const isLvn = vp.lvn.some(l => Math.abs(l.price - level.price) / level.price < 0.005);

                          return (
                            <div key={i} className={`vp-row ${level.is_poc ? 'poc' : ''} ${isInValueArea ? 'va' : ''} ${isCurrentPrice ? 'current-price' : ''}`}>
                              <span className="vp-price">{formatPrice(level.price)}</span>
                              <div className="vp-bar-container">
                                <div
                                  className={`vp-bar ${level.is_poc ? 'poc' : isInValueArea ? 'va' : 'normal'}`}
                                  style={{ width: `${Math.max(2, level.volume_relative)}%` }}
                                >
                                  {level.is_poc && <span className="vp-bar-label">POC</span>}
                                </div>
                                {isCurrentPrice && <div className="vp-current-marker">◄ Current</div>}
                              </div>
                              <span className="vp-vol-pct">{level.volume_pct}%</span>
                              {isHvn && <Tooltip text="High Volume Node — a price level where heavy trading occurred. Acts as strong support or resistance. Price tends to consolidate here."><span className="vp-node-badge hvn">HVN</span></Tooltip>}
                              {isLvn && <Tooltip text="Low Volume Node — a price level with little trading activity. Price moves quickly through these zones. Potential breakout/breakdown areas."><span className="vp-node-badge lvn">LVN</span></Tooltip>}
                            </div>
                          );
                        })}
                      </div>

                      {analysis.volume_analysis.volume_profile.analysis && (
                        <div className="vp-analysis-text">
                          {analysis.volume_analysis.volume_profile.analysis}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="volume-grid">
                    {/* OBV Card */}
                    <div className="volume-card">
                      <div className="volume-card-header">
                        <Tooltip text="On-Balance Volume tracks cumulative buying/selling pressure. ACCUMULATION = buyers dominating (bullish). DISTRIBUTION = sellers dominating (bearish). Divergence from price signals potential reversal.">
                          <span className="volume-card-title">OBV Trend</span>
                        </Tooltip>
                        <span className="volume-card-badge" style={{ color: getSignalColor(analysis.volume_analysis.obv.trend) }}>
                          {analysis.volume_analysis.obv.trend}
                        </span>
                      </div>
                      <div className="volume-card-body">
                        <div className="volume-stat">
                          <span className="stat-label">Current OBV</span>
                          <span className="stat-value">{(analysis.volume_analysis.obv.current / 1e6).toFixed(2)}M</span>
                        </div>
                        {analysis.volume_analysis.obv.divergence !== 'NONE' && (
                          <div className="volume-divergence-alert">
                            ⚠️ {analysis.volume_analysis.obv.divergence} divergence detected
                          </div>
                        )}
                      </div>
                    </div>

                    {/* MFI Card */}
                    <div className="volume-card">
                      <div className="volume-card-header">
                        <Tooltip text="Money Flow Index (0-100) — like RSI but incorporates volume. Above 80 = overbought (sell pressure likely). Below 20 = oversold (buy opportunity). Uses price and volume together for stronger signals.">
                          <span className="volume-card-title">Money Flow (MFI)</span>
                        </Tooltip>
                        <span className="volume-card-badge" style={{ color: getSignalColor(analysis.volume_analysis.mfi.zone) }}>
                          {analysis.volume_analysis.mfi.zone}
                        </span>
                      </div>
                      <div className="volume-card-body">
                        <div className="mfi-meter">
                          <div className="mfi-bar">
                            <div
                              className="mfi-fill"
                              style={{ width: `${analysis.volume_analysis.mfi.value}%` }}
                            ></div>
                            <div
                              className="mfi-pointer"
                              style={{ left: `${analysis.volume_analysis.mfi.value}%` }}
                            ></div>
                          </div>
                          <div className="mfi-labels">
                            <span>0</span>
                            <span>Oversold</span>
                            <span>Neutral</span>
                            <span>Overbought</span>
                            <span>100</span>
                          </div>
                        </div>
                        <div className="mfi-value-display">
                          <span className="mfi-number">{analysis.volume_analysis.mfi.value.toFixed(1)}</span>
                          <span className="mfi-interp">{analysis.volume_analysis.mfi.interpretation}</span>
                        </div>
                      </div>
                    </div>

                    {/* Buy/Sell Delta Card */}
                    <div className="volume-card">
                      <div className="volume-card-header">
                        <Tooltip text="Difference between buying and selling volume over the last 24 candles. Positive = more buying pressure. Strength indicates how one-sided the flow is (STRONG/MODERATE/WEAK).">
                          <span className="volume-card-title">Buy/Sell Delta</span>
                        </Tooltip>
                        <span className="volume-card-badge" style={{ color: getSignalColor(analysis.volume_analysis.buy_sell_delta.pressure) }}>
                          {analysis.volume_analysis.buy_sell_delta.pressure}
                        </span>
                      </div>
                      <div className="volume-card-body">
                        <div className="delta-display">
                          <span className={`delta-value ${analysis.volume_analysis.buy_sell_delta.delta_pct >= 0 ? 'positive' : 'negative'}`}>
                            {analysis.volume_analysis.buy_sell_delta.delta_pct >= 0 ? '+' : ''}{analysis.volume_analysis.buy_sell_delta.delta_pct.toFixed(1)}%
                          </span>
                          <span className="delta-strength">Strength: {analysis.volume_analysis.buy_sell_delta.strength}</span>
                        </div>
                        <div className="delta-bar-container">
                          <span className="delta-label sell">Sell</span>
                          <div className="delta-bar">
                            <div className="delta-buy" style={{ width: `${Math.max(0, 50 + analysis.volume_analysis.buy_sell_delta.delta_pct / 2)}%` }}></div>
                          </div>
                          <span className="delta-label buy">Buy</span>
                        </div>
                      </div>
                    </div>

                    {/* Volume Spikes Card */}
                    <div className="volume-card">
                      <div className="volume-card-header">
                        <Tooltip text="Compares current volume to the 20-period average. Above 1.5x = spike (unusual activity). Spikes often precede breakouts or signal institutional activity.">
                          <span className="volume-card-title">Volume Spikes</span>
                        </Tooltip>
                        {analysis.volume_analysis.volume_spikes.is_spike && (
                          <span className="spike-alert">🔥 SPIKE</span>
                        )}
                      </div>
                      <div className="volume-card-body">
                        <div className="volume-stat">
                          <span className="stat-label">Current vs Average</span>
                          <span className={`stat-value ${analysis.volume_analysis.volume_spikes.current_ratio >= 2 ? 'spike' : ''}`}>
                            {analysis.volume_analysis.volume_spikes.current_ratio.toFixed(2)}x
                          </span>
                        </div>
                        <div className="volume-stat">
                          <span className="stat-label">Spikes (24h)</span>
                          <span className="stat-value">{analysis.volume_analysis.volume_spikes.spike_count_24h}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </section>
              )}

              {/* Market Sentiment Section */}
              <section className="sentiment-section">
                <h2>🌡️ Market Sentiment</h2>
                {sentimentLoading ? (
                  <div className="sentiment-loading">
                    <div className="ai-spinner"></div>
                    <p>Loading market sentiment...</p>
                  </div>
                ) : marketSentiment ? (
                  <>
                    <div className={`sentiment-banner ${marketSentiment.overall_sentiment.toLowerCase()}`}>
                      <Tooltip text="Combined sentiment from Fear & Greed Index, Funding Rates, and Open Interest. Reflects the overall market mood and positioning.">
                        <span className="sentiment-label">Overall Market Sentiment</span>
                      </Tooltip>
                      <span className="sentiment-value" style={{ color: getSignalColor(marketSentiment.overall_sentiment) }}>
                        {marketSentiment.overall_sentiment}
                      </span>
                    </div>

                    <div className="sentiment-grid">
                      {/* Fear & Greed */}
                      <div className="sentiment-card fear-greed-card">
                        <div className="sentiment-card-header">
                          <Tooltip text="Crypto market sentiment index (0-100) from alternative.me. 0-24 = Extreme Fear (buy opportunity), 25-49 = Fear, 50-74 = Greed, 75-100 = Extreme Greed (sell signal). Contrarian indicator.">
                            <span className="sentiment-card-title">Fear & Greed Index</span>
                          </Tooltip>
                        </div>
                        <div className="sentiment-card-body">
                          {marketSentiment.fear_greed.error ? (
                            <div className="sentiment-error">{marketSentiment.fear_greed.error}</div>
                          ) : (
                            <>
                              <div className="fear-greed-gauge">
                                <div className="fg-bar">
                                  <div className="fg-pointer" style={{ left: `${marketSentiment.fear_greed.value}%` }}>
                                    <span className="fg-pointer-value">{marketSentiment.fear_greed.value}</span>
                                  </div>
                                </div>
                                <div className="fg-labels">
                                  <span style={{ color: '#ef4444' }}>Extreme Fear</span>
                                  <span style={{ color: '#f59e0b' }}>Fear</span>
                                  <span style={{ color: '#6b7280' }}>Neutral</span>
                                  <span style={{ color: '#22c55e' }}>Greed</span>
                                  <span style={{ color: '#10b981' }}>Extreme Greed</span>
                                </div>
                              </div>
                              <div className="fg-value-display">
                                <span className="fg-number" style={{ color: getFearGreedColor(marketSentiment.fear_greed.value) }}>
                                  {marketSentiment.fear_greed.value}
                                </span>
                                <span className="fg-label-text">{marketSentiment.fear_greed.label}</span>
                                {marketSentiment.fear_greed.trend && (
                                  <span className="fg-trend">Trend: {marketSentiment.fear_greed.trend}</span>
                                )}
                              </div>
                            </>
                          )}
                        </div>
                      </div>

                      {/* Funding Rate */}
                      <div className="sentiment-card">
                        <div className="sentiment-card-header">
                          <Tooltip text="Perpetual futures funding rate. Positive = longs pay shorts (market is bullish). Negative = shorts pay longs (market is bearish). Extreme rates often precede reversals.">
                            <span className="sentiment-card-title">Funding Rate</span>
                          </Tooltip>
                          {!marketSentiment.funding_rate.error && (
                            <span className="sentiment-card-badge" style={{ color: getSignalColor(marketSentiment.funding_rate.sentiment) }}>
                              {marketSentiment.funding_rate.sentiment}
                            </span>
                          )}
                        </div>
                        <div className="sentiment-card-body">
                          {marketSentiment.funding_rate.error ? (
                            <div className="sentiment-error">{marketSentiment.funding_rate.error}</div>
                          ) : (
                            <>
                              <div className={`funding-rate-value ${marketSentiment.funding_rate.rate >= 0 ? 'positive' : 'negative'}`}>
                                {marketSentiment.funding_rate.rate_pct}
                              </div>
                              <div className="funding-interp">{marketSentiment.funding_rate.interpretation}</div>
                            </>
                          )}
                        </div>
                      </div>

                      {/* Open Interest */}
                      <div className="sentiment-card">
                        <div className="sentiment-card-header">
                          <Tooltip text="Total value of all open futures contracts. Rising OI + rising price = strong bullish trend. Rising OI + falling price = strong bearish trend. Dropping OI = positions closing.">
                            <span className="sentiment-card-title">Open Interest</span>
                          </Tooltip>
                        </div>
                        <div className="sentiment-card-body">
                          {marketSentiment.open_interest.error ? (
                            <div className="sentiment-error">{marketSentiment.open_interest.error}</div>
                          ) : (
                            <div className="oi-display">
                              <span className="oi-value">{marketSentiment.open_interest.formatted}</span>
                              <span className="oi-label">Total Open Contracts</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="sentiment-empty">
                    <p>Run analysis to load market sentiment data</p>
                  </div>
                )}
              </section>

              {/* Derivatives Intelligence Section */}
              <section className="derivatives-section">
                <h2>📊 Derivatives Intelligence</h2>
                {derivativesLoading ? (
                  <div className="derivatives-loading">
                    <div className="ai-spinner"></div>
                    <p>Loading derivatives data...</p>
                  </div>
                ) : derivativesData ? (
                  <>
                    <div className={`derivatives-banner ${derivativesData.overall_signal.toLowerCase()}`}>
                      <Tooltip text="Composite signal from Long/Short ratios, Taker volume, Order Book depth, and Liquidation analysis. Indicates whether derivatives traders are positioned bullish or bearish.">
                        <span className="derivatives-label">Derivatives Signal</span>
                      </Tooltip>
                      <span className="derivatives-value" style={{ color: getSignalColor(derivativesData.overall_signal) }}>
                        {derivativesData.overall_signal}
                      </span>
                    </div>

                    {/* Long/Short Ratio */}
                    <div className="deriv-subsection">
                      <Tooltip text="Ratio of long vs short positions on Binance Futures. Top Traders = professional/whale accounts. Global = all users. Divergence between them reveals 'smart money' positioning.">
                        <h3>Long/Short Ratio</h3>
                      </Tooltip>
                      <div className="ls-grid">
                        <div className="ls-card">
                          <div className="ls-card-header">Top Traders</div>
                          <div className="ls-bar-container">
                            <div className="ls-bar-long" style={{ width: `${derivativesData.long_short_ratio.top_traders.long_pct}%` }}>
                              {derivativesData.long_short_ratio.top_traders.long_pct.toFixed(1)}% L
                            </div>
                            <div className="ls-bar-short" style={{ width: `${derivativesData.long_short_ratio.top_traders.short_pct}%` }}>
                              {derivativesData.long_short_ratio.top_traders.short_pct.toFixed(1)}% S
                            </div>
                          </div>
                          <div className="ls-trend">{derivativesData.long_short_ratio.top_traders.trend.replace(/_/g, ' ')}</div>
                        </div>
                        <div className="ls-card">
                          <div className="ls-card-header">Global Accounts</div>
                          <div className="ls-bar-container">
                            <div className="ls-bar-long" style={{ width: `${derivativesData.long_short_ratio.global.long_pct}%` }}>
                              {derivativesData.long_short_ratio.global.long_pct.toFixed(1)}% L
                            </div>
                            <div className="ls-bar-short" style={{ width: `${derivativesData.long_short_ratio.global.short_pct}%` }}>
                              {derivativesData.long_short_ratio.global.short_pct.toFixed(1)}% S
                            </div>
                          </div>
                          <div className="ls-trend">{derivativesData.long_short_ratio.global.trend.replace(/_/g, ' ')}</div>
                        </div>
                      </div>
                      {derivativesData.long_short_ratio.signal !== 'NEUTRAL' && (
                        <div className={`smart-money-alert ${derivativesData.long_short_ratio.signal.includes('LONG') ? 'bullish' : 'bearish'}`}>
                          💡 {derivativesData.long_short_ratio.signal.replace(/_/g, ' ')}
                        </div>
                      )}
                    </div>

                    {/* Order Flow */}
                    <div className="deriv-subsection">
                      <Tooltip text="Real-time order flow analysis. Taker Buy/Sell = aggressive market orders (shows urgency). Order Book Depth = resting limit orders (shows where support/resistance walls sit).">
                        <h3>Order Flow</h3>
                      </Tooltip>
                      <div className="orderflow-grid">
                        <div className="orderflow-card">
                          <div className="orderflow-card-header">Taker Buy/Sell</div>
                          <div className="taker-bar-container">
                            <div className="taker-bar-buy" style={{ width: `${(derivativesData.taker_volume.ratio / (derivativesData.taker_volume.ratio + 1)) * 100}%` }}>
                              Buy
                            </div>
                            <div className="taker-bar-sell" style={{ width: `${(1 / (derivativesData.taker_volume.ratio + 1)) * 100}%` }}>
                              Sell
                            </div>
                          </div>
                          <div className="orderflow-stats">
                            <span>Ratio: {derivativesData.taker_volume.ratio.toFixed(3)}</span>
                            <span className={`pressure-badge ${derivativesData.taker_volume.pressure.toLowerCase().replace('strong_', '')}`}>
                              {derivativesData.taker_volume.pressure.replace(/_/g, ' ')}
                            </span>
                          </div>
                        </div>
                        <div className="orderflow-card">
                          <div className="orderflow-card-header">Order Book Depth</div>
                          <div className="ob-bar-container">
                            <div className="ob-bar-bid" style={{ width: `${(derivativesData.order_book.bid_ask_ratio / (derivativesData.order_book.bid_ask_ratio + 1)) * 100}%` }}>
                              Bids
                            </div>
                            <div className="ob-bar-ask" style={{ width: `${(1 / (derivativesData.order_book.bid_ask_ratio + 1)) * 100}%` }}>
                              Asks
                            </div>
                          </div>
                          <div className="orderflow-stats">
                            <span>Ratio: {derivativesData.order_book.bid_ask_ratio.toFixed(3)}</span>
                            <span className={`imbalance-badge ${derivativesData.order_book.imbalance.toLowerCase().includes('bid') ? 'bid' : derivativesData.order_book.imbalance.toLowerCase().includes('ask') ? 'ask' : 'balanced'}`}>
                              {derivativesData.order_book.imbalance.replace(/_/g, ' ')}
                            </span>
                          </div>
                          <div className="ob-walls">
                            <div className="ob-wall bid-wall">
                              <span className="wall-label">Support Wall</span>
                              <span className="wall-price">{formatPrice(derivativesData.order_book.strongest_bid.price)}</span>
                              <span className="wall-size">${(derivativesData.order_book.strongest_bid.size_usd / 1000).toFixed(0)}K</span>
                            </div>
                            <div className="ob-wall ask-wall">
                              <span className="wall-label">Resistance Wall</span>
                              <span className="wall-price">{formatPrice(derivativesData.order_book.strongest_ask.price)}</span>
                              <span className="wall-size">${(derivativesData.order_book.strongest_ask.size_usd / 1000).toFixed(0)}K</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Liquidation Map */}
                    <div className="deriv-subsection">
                      <Tooltip text="Estimated price levels where leveraged positions get liquidated. Cascading liquidations create rapid price moves. Closer levels (high leverage) get hit first.">
                        <h3>Liquidation Map</h3>
                      </Tooltip>
                      {derivativesData.liquidations.oi_change_pct !== 0 && (
                        <div className={`oi-change-banner ${derivativesData.liquidations.oi_change_pct > 0 ? 'positive' : 'negative'}`}>
                          OI 24h: {derivativesData.liquidations.oi_change_pct > 0 ? '+' : ''}{derivativesData.liquidations.oi_change_pct.toFixed(1)}%
                          {derivativesData.liquidations.recent_signal !== 'NONE' && (
                            <span className="liq-signal"> — {derivativesData.liquidations.recent_signal.replace(/_/g, ' ')}</span>
                          )}
                        </div>
                      )}
                      <div className="liq-map">
                        <div className="liq-side liq-longs">
                          <div className="liq-side-header">Long Liquidations (Below)</div>
                          {derivativesData.liquidations.long_liq_levels.map((level, idx) => (
                            <div key={idx} className="liq-level long-level">
                              <span className="liq-leverage">{level.leverage}</span>
                              <div className="liq-bar-track">
                                <div className="liq-bar long-bar" style={{ width: `${Math.min(level.distance_pct * 5, 100)}%` }}></div>
                              </div>
                              <span className="liq-price">{formatPrice(level.price)}</span>
                              <span className="liq-dist">-{level.distance_pct.toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                        <div className="liq-side liq-shorts">
                          <div className="liq-side-header">Short Liquidations (Above)</div>
                          {derivativesData.liquidations.short_liq_levels.map((level, idx) => (
                            <div key={idx} className="liq-level short-level">
                              <span className="liq-leverage">{level.leverage}</span>
                              <div className="liq-bar-track">
                                <div className="liq-bar short-bar" style={{ width: `${Math.min(level.distance_pct * 5, 100)}%` }}></div>
                              </div>
                              <span className="liq-price">{formatPrice(level.price)}</span>
                              <span className="liq-dist">+{level.distance_pct.toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="derivatives-empty">
                    <p>Run analysis to load derivatives intelligence</p>
                  </div>
                )}
              </section>

              {/* Whale Activity Section */}
              <section className="whale-section">
                <h2>🐋 Whale Activity</h2>
                {whaleLoading ? (
                  <div className="whale-loading">
                    <div className="ai-spinner"></div>
                    <p>Scanning for whale activity...</p>
                  </div>
                ) : whaleData ? (
                  <>
                    <div className={`whale-banner ${whaleData.whale_signal.toLowerCase()}`}>
                      <Tooltip text="Tracks large transactions and unusual trade sizes. ACCUMULATION = big buyers active (bullish). DISTRIBUTION = big sellers active (bearish). ACTIVE = significant whale movement detected.">
                        <span className="whale-label">Whale Signal</span>
                      </Tooltip>
                      <span className="whale-value" style={{
                        color: whaleData.whale_signal === 'ACCUMULATION' ? '#00e676' :
                               whaleData.whale_signal === 'DISTRIBUTION' ? '#ff5252' :
                               whaleData.whale_signal === 'ACTIVE' ? '#ffab40' : '#888'
                      }}>
                        {whaleData.whale_signal}
                      </span>
                    </div>

                    {whaleData.alert && (
                      <div className="whale-alert">
                        ⚠️ {whaleData.alert}
                      </div>
                    )}

                    {whaleData.recent_large_txs.length > 0 ? (
                      <div className="whale-txs">
                        <div className="whale-txs-header">
                          <span>Recent Large Transactions</span>
                          <span className="whale-source">{whaleData.source}</span>
                        </div>
                        {whaleData.recent_large_txs.map((tx, idx) => (
                          <div key={idx} className="whale-tx-row">
                            <span className="tx-time">{tx.time}</span>
                            <span className="tx-value">{tx.value_display}</span>
                            {tx.type && (
                              <span className={`tx-type ${tx.type.toLowerCase()}`}>{tx.type}</span>
                            )}
                            {tx.hash && (
                              <span className="tx-hash">{tx.hash}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="whale-empty-txs">
                        <p>No large transactions detected in recent window</p>
                        {whaleData.large_trade_count > 0 && (
                          <p className="whale-trade-stats">
                            {whaleData.large_trade_count} outlier trades found | Avg size: ${whaleData.avg_large_trade_size.toLocaleString()}
                          </p>
                        )}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="whale-empty">
                    <p>Run analysis to scan whale activity</p>
                  </div>
                )}
              </section>

              {/* Active Scenarios */}
              {analysis.active_scenarios.length > 0 && (
                <section className="scenarios-section">
                  <h2>⚠️ Active Scenarios ({analysis.scenario_count})</h2>
                  <p className="section-desc">These conditions are affecting your analysis</p>
                  
                  <div className="scenarios-grid">
                    {analysis.active_scenarios.map((scenario, idx) => (
                      <div 
                        key={idx}
                        className="scenario-card"
                        style={{ borderLeftColor: getSeverityColor(scenario.severity) }}
                      >
                        <div className="scenario-header">
                          <span className="scenario-icon">{scenario.icon}</span>
                          <span className="scenario-title">{scenario.title}</span>
                          <span 
                            className="severity-badge"
                            style={{ backgroundColor: getSeverityColor(scenario.severity) }}
                          >
                            {scenario.severity}
                          </span>
                        </div>
                        <div className="scenario-message">{scenario.message}</div>
                        {scenario.details && (
                          <div className="scenario-details">{scenario.details}</div>
                        )}
                        <div className="scenario-effect">Effect: {scenario.effect}</div>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* Trade Type Requirements */}
              {analysis.trade_type_requirements && analysis.trade_type_requirements.length > 0 && (
                <section className="requirements-section">
                  <h2>📋 {analysis.trade_type_info?.name || 'Trade'} Requirements</h2>
                  <p className="section-desc">
                    Requirements for {analysis.trade_type_info?.name} trading with {analysis.experience_info?.name} experience
                  </p>
                  
                  <div className="requirements-grid">
                    {analysis.trade_type_requirements.map((req, idx) => (
                      <div key={idx} className={`requirement-card ${req.met ? 'met' : 'not-met'}`}>
                        <div className="req-header">
                          <span className="req-name">{req.name}</span>
                          <span className={`req-status ${req.met ? 'pass' : 'fail'}`}>
                            {req.met ? '✅' : '❌'}
                          </span>
                        </div>
                        <div className="req-values">
                          <div className="req-required">
                            <span className="req-label">Required:</span>
                            <span className="req-value">{req.required}</span>
                          </div>
                          <div className="req-current">
                            <span className="req-label">Current:</span>
                            <span className={`req-value ${req.met ? 'good' : 'bad'}`}>{req.current}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Trade Type Info Box */}
                  {analysis.trade_type_info && (
                    <div className="trade-type-summary">
                      <div className="tts-header">
                        <span className="tts-name">{analysis.trade_type_info.name} Trading</span>
                        <span className={`tts-risk risk-${analysis.trade_type_info.risk_level.toLowerCase().replace(' ', '-')}`}>
                          {analysis.trade_type_info.risk_level} Risk
                        </span>
                      </div>
                      <div className="tts-duration">⏱️ {analysis.trade_type_info.duration_display}</div>
                      <div className="tts-desc">{analysis.trade_type_info.description}</div>
                      <div className="tts-stats">
                        <span>Base Threshold: {(analysis.trade_type_info.base_threshold * 100).toFixed(0)}%</span>
                        <span>Min ADX: {analysis.trade_type_info.min_adx || 'None'}</span>
                        <span>Min Expectancy: {analysis.trade_type_info.min_expectancy}%</span>
                      </div>
                    </div>
                  )}
                </section>
              )}

              {/* Forecast Section */}
              <section className="forecast-section">
                <h2>🔮 Price Forecast</h2>
                
                <Tooltip text="The ML model's directional forecast based on multi-timeframe analysis. BULLISH = expects price increase, BEARISH = expects decrease, SIDEWAYS = no clear direction.">
                  <div className={`forecast-direction ${analysis.forecast.direction.toLowerCase()}`}>
                    Outlook: {analysis.forecast.direction}
                  </div>
                </Tooltip>

                <div className="forecast-probs">
                  <div className="forecast-prob up">
                    <span className="fp-icon">📈</span>
                    <Tooltip text="Probability of price moving up beyond the positive threshold (typically +3%) within the forecast window for your trade type.">
                      <span className="fp-label">Upside</span>
                    </Tooltip>
                    <span className="fp-value">{analysis.forecast.probabilities.up}%</span>
                  </div>
                  <div className="forecast-prob sideways">
                    <span className="fp-icon">➡️</span>
                    <Tooltip text="Probability of price staying within the -3% to +3% range. High sideways probability means range-bound, choppy conditions.">
                      <span className="fp-label">Sideways</span>
                    </Tooltip>
                    <span className="fp-value">{analysis.forecast.probabilities.sideways}%</span>
                  </div>
                  <div className="forecast-prob down">
                    <span className="fp-icon">📉</span>
                    <Tooltip text="Probability of price dropping below the negative threshold (typically -3%). High downside probability is a warning to avoid or hedge.">
                      <span className="fp-label">Downside</span>
                    </Tooltip>
                    <span className="fp-value">{analysis.forecast.probabilities.down}%</span>
                  </div>
                </div>

                <div className="price-targets">
                  <div className="target bull">
                    <Tooltip text="The optimistic price target if the upside scenario plays out. Based on recent volatility and historical move patterns.">
                      <span className="target-label">Bull Target</span>
                    </Tooltip>
                    <span className="target-price">{formatPrice(analysis.forecast.bull_target)}</span>
                  </div>
                  <div className="target current">
                    <span className="target-label">Current</span>
                    <span className="target-price">{formatPrice(analysis.forecast.current_price)}</span>
                  </div>
                  <div className="target bear">
                    <Tooltip text="The pessimistic price target if the downside scenario plays out. Useful for setting stop-loss levels.">
                      <span className="target-label">Bear Target</span>
                    </Tooltip>
                    <span className="target-price">{formatPrice(analysis.forecast.bear_target)}</span>
                  </div>
                </div>
              </section>

              {/* Risk Management (only for BUY) */}
              {analysis.risk.action === 'OPEN_POSITION' && (
                <section className="risk-section">
                  <h2>⚖️ Position Sizing</h2>
                  
                  <div className="risk-grid">
                    <div className="risk-card">
                      <Tooltip text="The recommended dollar amount to invest based on your capital, trade type risk rules, and the Kelly Criterion. Never risk more than this on a single trade.">
                        <span className="risk-label">Position Size</span>
                      </Tooltip>
                      <span className="risk-value">${analysis.risk.position_size_usd}</span>
                      <span className="risk-pct">{analysis.risk.position_size_pct}% of capital</span>
                    </div>
                    <div className="risk-card">
                      <Tooltip text="The recommended price to enter the trade. Usually the current market price at analysis time.">
                        <span className="risk-label">Entry Price</span>
                      </Tooltip>
                      <span className="risk-value">{formatPrice(analysis.risk.entry_price || 0)}</span>
                    </div>
                    <div className="risk-card loss">
                      <Tooltip text="Exit the trade if price drops to this level. Limits your maximum loss. Set based on ATR (volatility) and trade type. ALWAYS use a stop loss.">
                        <span className="risk-label">Stop Loss</span>
                      </Tooltip>
                      <span className="risk-value">{formatPrice(analysis.risk.stop_loss_price || 0)}</span>
                      <span className="risk-pct">-{analysis.risk.stop_loss_pct}%</span>
                    </div>
                    <div className="risk-card win">
                      <Tooltip text="Target price to take profits. Set based on risk/reward ratio and volatility. Consider taking partial profits here.">
                        <span className="risk-label">Take Profit</span>
                      </Tooltip>
                      <span className="risk-value">{formatPrice(analysis.risk.take_profit_price || 0)}</span>
                      <span className="risk-pct">+{analysis.risk.take_profit_pct}%</span>
                    </div>
                  </div>

                  <div className="risk-summary">
                    <div className="max-loss">
                      <Tooltip text="The worst-case dollar loss if your stop loss is hit. This is the actual money at risk on this trade.">
                        <span>Maximum Loss: <strong>${analysis.risk.max_loss_usd}</strong></span>
                      </Tooltip>
                    </div>
                    <div className="hold-time">
                      <Tooltip text="Don't hold the trade longer than this. Time-based exit prevents capital from being tied up in stale positions.">
                        <span>Max Hold Time: <strong>{analysis.risk.max_hold_hours}h</strong></span>
                      </Tooltip>
                    </div>
                  </div>
                </section>
              )}

              {/* Reasoning Section */}
              <section className="reasoning-section">
                <h2>💭 Analysis Reasoning</h2>
                <ul className="reasoning-list">
                  {analysis.reasoning.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </section>

              {/* AI Analysis Section */}
              <section className="ai-section">
                <div className="ai-header">
                  <h2>🤖 AI-Powered Insights</h2>
                  <button
                    className="ai-analyze-btn"
                    onClick={fetchAIAnalysis}
                    disabled={aiLoading || !capital || capital <= 0}
                  >
                    {aiLoading ? '🔄 AI Analyzing...' : '🧠 Run Deep Analysis'}
                  </button>
                </div>

                {!aiAvailable && !aiAnalysis && (
                  <div className="ai-notice">
                    AI providers not configured. Add GROQ_API_KEY to your .env file (free at console.groq.com).
                  </div>
                )}

                {aiLoading && (
                  <div className="ai-loading">
                    <div className="ai-spinner"></div>
                    <p>AI is cross-referencing technicals, volume, sentiment & news...</p>
                  </div>
                )}

                {aiAnalysis && !aiLoading && (
                  <>
                    {aiAnalysis.error ? (
                      <div className="ai-error">
                        <span className="ai-error-icon">⚠️</span>
                        <span>{aiAnalysis.error}</span>
                      </div>
                    ) : (
                      <div className="ai-results">
                        {/* Verdict Bar */}
                        <div className="ai-verdict-row">
                          <div className={`ai-verdict-badge ${(aiAnalysis.ai_verdict || '').toLowerCase()}`}>
                            {aiAnalysis.ai_verdict}
                          </div>
                          <div className="ai-confidence">
                            <Tooltip text="The AI's self-assessed confidence in its verdict (1-10). Based on how aligned all data points are. 8+ = high conviction, 5-7 = moderate, below 5 = conflicting signals.">
                              <span className="ai-conf-label">Confidence</span>
                            </Tooltip>
                            <div className="ai-conf-bar">
                              <div
                                className="ai-conf-fill"
                                style={{ width: `${(aiAnalysis.confidence_score || 0) * 10}%` }}
                              ></div>
                            </div>
                            <span className="ai-conf-value">{aiAnalysis.confidence_score}/10</span>
                          </div>
                          <div className={`ai-sentiment ${(aiAnalysis.sentiment_read || '').toLowerCase()}`}>
                            {aiAnalysis.sentiment_read === 'BULLISH' ? '📈' : aiAnalysis.sentiment_read === 'BEARISH' ? '📉' : '➡️'} {aiAnalysis.sentiment_read}
                          </div>
                        </div>

                        {/* TLDR */}
                        {(aiAnalysis.tldr || aiAnalysis.summary) && (
                          <div className="ai-tldr">
                            <Tooltip text="A 2-3 sentence executive summary of the AI's complete analysis. Captures the key situation, risk/reward, and recommended action.">
                              <div className="ai-tldr-label">TLDR</div>
                            </Tooltip>
                            <p>{aiAnalysis.tldr || aiAnalysis.summary}</p>
                          </div>
                        )}

                        {/* Key Insights */}
                        {aiAnalysis.key_insights && aiAnalysis.key_insights.length > 0 && (
                          <div className="ai-insights-enhanced">
                            <h4>{coin.replace('_', '/')} Insights</h4>
                            <div className="ai-insights-list">
                              {aiAnalysis.key_insights.map((insight, i) => (
                                <div key={i} className="ai-insight-item">
                                  <span className="ai-insight-number">{i + 1}</span>
                                  <span className="ai-insight-text">{insight}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Positives & Risks Side by Side */}
                        <div className="ai-pos-risk-grid">
                          {/* Positives */}
                          {aiAnalysis.positives && aiAnalysis.positives.length > 0 && (
                            <div className="ai-pos-section">
                              <Tooltip text="Bullish factors the AI identified by cross-referencing technicals, volume, sentiment, derivatives, and news data.">
                                <h4 className="ai-pos-header">Positives</h4>
                              </Tooltip>
                              <div className="ai-pos-list">
                                {aiAnalysis.positives.map((pos, i) => (
                                  <div key={i} className="ai-pos-item">
                                    <div className="ai-pos-item-header">
                                      <span className="ai-pos-number">{i + 1}.</span>
                                      <span className="ai-pos-title">{pos.title}</span>
                                    </div>
                                    <p className="ai-pos-detail">{pos.detail}</p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Risks */}
                          {aiAnalysis.risks && aiAnalysis.risks.length > 0 && (
                            <div className="ai-risk-section">
                              <Tooltip text="Risk factors and concerns. HIGH severity = deal-breakers, MEDIUM = significant but manageable, LOW = minor concerns to monitor.">
                                <h4 className="ai-risk-header">Risks</h4>
                              </Tooltip>
                              <div className="ai-risk-list">
                                {aiAnalysis.risks.map((risk, i) => (
                                  <div key={i} className={`ai-risk-item severity-${(risk.severity || 'medium').toLowerCase()}`}>
                                    <div className="ai-risk-item-header">
                                      <span className="ai-risk-number">{i + 1}.</span>
                                      <span className="ai-risk-title">{risk.title}</span>
                                      <span className={`ai-risk-severity ${(risk.severity || '').toLowerCase()}`}>
                                        {risk.severity}
                                      </span>
                                    </div>
                                    <p className="ai-risk-detail">{risk.detail}</p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>

                        {/* Market Pulse */}
                        {aiAnalysis.market_pulse && (
                          <div className="ai-market-pulse">
                            <Tooltip text="Quick snapshot of four key market dimensions: trend direction, volume activity, market sentiment, and macro/BTC conditions.">
                              <h4>Market Pulse</h4>
                            </Tooltip>
                            <div className="ai-pulse-grid">
                              <div className="ai-pulse-item">
                                <span className="ai-pulse-icon">📊</span>
                                <span className="ai-pulse-label">Trend</span>
                                <p>{aiAnalysis.market_pulse.trend_summary}</p>
                              </div>
                              <div className="ai-pulse-item">
                                <span className="ai-pulse-icon">📦</span>
                                <span className="ai-pulse-label">Volume</span>
                                <p>{aiAnalysis.market_pulse.volume_verdict}</p>
                              </div>
                              <div className="ai-pulse-item">
                                <span className="ai-pulse-icon">🧠</span>
                                <span className="ai-pulse-label">Sentiment</span>
                                <p>{aiAnalysis.market_pulse.sentiment_verdict}</p>
                              </div>
                              <div className="ai-pulse-item">
                                <span className="ai-pulse-icon">🌐</span>
                                <span className="ai-pulse-label">Macro</span>
                                <p>{aiAnalysis.market_pulse.macro_context}</p>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Entry Strategy */}
                        {aiAnalysis.entry_strategy && (
                          <div className="ai-entry-strategy">
                            <Tooltip text="Specific, actionable trading plan with entry conditions, price levels for stops and targets. Follow this if you decide to trade.">
                              <h4>🎯 Entry Strategy</h4>
                            </Tooltip>
                            <p>{aiAnalysis.entry_strategy}</p>
                          </div>
                        )}

                        {/* What to Watch */}
                        {aiAnalysis.what_to_watch && aiAnalysis.what_to_watch.length > 0 && (
                          <div className="ai-watch">
                            <Tooltip text="Key triggers, price levels, or events that could change the trade thesis. Monitor these before and after entry.">
                              <h4>👀 What to Watch</h4>
                            </Tooltip>
                            <div className="ai-watch-items">
                              {aiAnalysis.what_to_watch.map((item, i) => (
                                <span key={i} className="ai-watch-tag">{item}</span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Conviction */}
                        {aiAnalysis.conviction_reason && (
                          <div className="ai-conviction">
                            <Tooltip text="The single most important factor driving the AI's verdict. If this changes, the entire analysis should be re-evaluated.">
                              <span className="ai-conviction-label">💡 Key Conviction</span>
                            </Tooltip>
                            <p>{aiAnalysis.conviction_reason}</p>
                          </div>
                        )}

                        {/* Provider Badge */}
                        <div className="ai-provider-badge">
                          Powered by {aiAnalysis.ai_provider} {aiAnalysis.ai_cost === 'FREE' ? '(FREE)' : ''}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </section>

              {/* Suggested Action */}
              <section className="action-section">
                <h2>🎯 Suggested Action</h2>
                <div className={`action-card ${analysis.suggested_action.action.toLowerCase()}`}>
                  <Tooltip text="EXECUTE = enter the trade now. MONITOR = conditions are close, watch for entry. STAY_OUT = wait for better setup. STOP = do not trade, conditions are unfavorable.">
                    <div className="action-name">{analysis.suggested_action.action}</div>
                  </Tooltip>
                  <div className="action-message">{analysis.suggested_action.message}</div>
                  {analysis.suggested_action.next_check && (
                    <Tooltip text="When to run the analysis again. Markets change — re-check at this interval to see if conditions have improved.">
                      <div className="next-check">⏰ Re-evaluate in: {analysis.suggested_action.next_check}</div>
                    </Tooltip>
                  )}
                  {analysis.suggested_action.conditions && analysis.suggested_action.conditions.length > 0 && (
                    <div className="conditions">
                      <strong>Conditions to watch:</strong>
                      <ul>
                        {analysis.suggested_action.conditions.filter(c => c).map((c, i) => (
                          <li key={i}>{c}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </section>
            </>
          ) : (
            <div className="empty-state">
              <p>Click "Run Full Analysis" to get started</p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default CoinPage;