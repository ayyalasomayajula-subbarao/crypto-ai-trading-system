import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Coinpage.css';

const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/prices';

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
  const navigate = useNavigate();
  const coin = coinId?.toUpperCase() || 'BTC_USDT';
  const coinConfig = COIN_CONFIG[coin] || COIN_CONFIG['BTC_USDT'];

  // State
  const [price, setPrice] = useState<PriceData | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [lastAnalysis, setLastAnalysis] = useState<Date | null>(null);

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
    } catch (error) {
      console.error('Error fetching analysis:', error);
    } finally {
      setLoading(false);
    }
  }, [coin, capital, tradeType, experience, reason, recentLosses, tradesToday]);

  // Initial fetch only if capital is set
  useEffect(() => {
    if (capital && capital > 0) {
      fetchAnalysis();
    }
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

  const tradeTypeInfo = TRADE_TYPE_INFO[tradeType];

  // ============================================================
  // RENDER
  // ============================================================
  return (
    <div className="coin-page">
      {/* Header */}
      <header className="coin-header">
        <button className="back-btn" onClick={() => navigate('/')}>
          ← Back to Dashboard
        </button>
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
                    <span className="verdict-label">Recommendation</span>
                    <span 
                      className="verdict-value"
                      style={{ color: getVerdictColor(analysis.verdict) }}
                    >
                      {analysis.verdict}
                    </span>
                    <span className="confidence-badge">{analysis.confidence} confidence</span>
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
                    <div className="regime-header">
                      <span>Market State</span>
                    </div>
                    <div className={`regime-badge ${analysis.market_context.regime.regime.toLowerCase()}`}>
                      {analysis.market_context.regime.regime}
                    </div>
                    <div className="volatility-info">
                      Volatility: <strong>{analysis.market_context.regime.volatility}</strong>
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
                
                <div className={`forecast-direction ${analysis.forecast.direction.toLowerCase()}`}>
                  Outlook: {analysis.forecast.direction}
                </div>

                <div className="forecast-probs">
                  <div className="forecast-prob up">
                    <span className="fp-icon">📈</span>
                    <span className="fp-label">Upside</span>
                    <span className="fp-value">{analysis.forecast.probabilities.up}%</span>
                  </div>
                  <div className="forecast-prob sideways">
                    <span className="fp-icon">➡️</span>
                    <span className="fp-label">Sideways</span>
                    <span className="fp-value">{analysis.forecast.probabilities.sideways}%</span>
                  </div>
                  <div className="forecast-prob down">
                    <span className="fp-icon">📉</span>
                    <span className="fp-label">Downside</span>
                    <span className="fp-value">{analysis.forecast.probabilities.down}%</span>
                  </div>
                </div>

                <div className="price-targets">
                  <div className="target bull">
                    <span className="target-label">Bull Target</span>
                    <span className="target-price">{formatPrice(analysis.forecast.bull_target)}</span>
                  </div>
                  <div className="target current">
                    <span className="target-label">Current</span>
                    <span className="target-price">{formatPrice(analysis.forecast.current_price)}</span>
                  </div>
                  <div className="target bear">
                    <span className="target-label">Bear Target</span>
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
                      <span className="risk-label">Position Size</span>
                      <span className="risk-value">${analysis.risk.position_size_usd}</span>
                      <span className="risk-pct">{analysis.risk.position_size_pct}% of capital</span>
                    </div>
                    <div className="risk-card">
                      <span className="risk-label">Entry Price</span>
                      <span className="risk-value">{formatPrice(analysis.risk.entry_price || 0)}</span>
                    </div>
                    <div className="risk-card loss">
                      <span className="risk-label">Stop Loss</span>
                      <span className="risk-value">{formatPrice(analysis.risk.stop_loss_price || 0)}</span>
                      <span className="risk-pct">-{analysis.risk.stop_loss_pct}%</span>
                    </div>
                    <div className="risk-card win">
                      <span className="risk-label">Take Profit</span>
                      <span className="risk-value">{formatPrice(analysis.risk.take_profit_price || 0)}</span>
                      <span className="risk-pct">+{analysis.risk.take_profit_pct}%</span>
                    </div>
                  </div>

                  <div className="risk-summary">
                    <div className="max-loss">
                      Maximum Loss: <strong>${analysis.risk.max_loss_usd}</strong>
                    </div>
                    <div className="hold-time">
                      Max Hold Time: <strong>{analysis.risk.max_hold_hours}h</strong>
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

              {/* Suggested Action */}
              <section className="action-section">
                <h2>🎯 Suggested Action</h2>
                <div className={`action-card ${analysis.suggested_action.action.toLowerCase()}`}>
                  <div className="action-name">{analysis.suggested_action.action}</div>
                  <div className="action-message">{analysis.suggested_action.message}</div>
                  {analysis.suggested_action.next_check && (
                    <div className="next-check">⏰ Re-evaluate in: {analysis.suggested_action.next_check}</div>
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