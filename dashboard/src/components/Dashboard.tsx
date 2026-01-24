import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import './Dashboard.css';

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

interface Signal {
  coin: string;
  price: number;
  verdict: string;
  confidence: string;
  win_probability: number | null;
  loss_probability: number | null;
  expectancy: number | null;
  readiness: number | null;
  scenario_count: number;
  model_ran: boolean;
}

interface BTCContext {
  trend_1h: string;
  trend_4h: string;
  trend_1d: string;
  overall_trend: string;
  support_alts: boolean;
  price: number;
  change_24h: number;
}

interface ActiveScenario {
  type: string;
  title: string;
  message: string;
  icon: string;
  severity: string;
  effect: string;
}

interface AnalysisResult {
  coin: string;
  timestamp: string;
  price: number;
  price_source: string;
  capital: number;
  trade_type: string;
  experience_level: string;
  verdict: string;
  confidence: string;
  model_ran: boolean;
  win_probability: number | null;
  loss_probability: number | null;
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
  };
  suggested_action: {
    action: string;
    message: string;
    next_check: string | null;
    conditions?: string[];
    why?: string;
  };
  market_context: {
    btc: BTCContext;
    regime: {
      regime: string;
      adx: number;
      volatility: string;
      recommendation: string;
    };
  };
}

interface ScanResponse {
  timestamp: string;
  market_context: {
    btc: BTCContext;
  };
  signals: Signal[];
  buy_signals: Signal[];
  blocked_count: number;
  market_summary: string;
}

// ============================================================
// MAIN DASHBOARD COMPONENT
// ============================================================

const Dashboard: React.FC = () => {
  // State
  const [prices, setPrices] = useState<Record<string, PriceData>>({});
  const [signals, setSignals] = useState<Signal[]>([]);
  const [btcContext, setBtcContext] = useState<BTCContext | null>(null);
  const [marketSummary, setMarketSummary] = useState<string>('');
  const [selectedCoin, setSelectedCoin] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // User inputs
  const [capital, setCapital] = useState(1000);
  const [tradeType, setTradeType] = useState('SWING');
  const [experience, setExperience] = useState('INTERMEDIATE');
  const [reason, setReason] = useState('');
  const [recentLosses, setRecentLosses] = useState(0);
  const [tradesToday, setTradesToday] = useState(0);

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null);

  // ============================================================
  // WEBSOCKET CONNECTION (Live Prices)
  // ============================================================
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setWsConnected(true);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'initial' || data.type === 'all_prices') {
          setPrices(data.prices);
        } else if (data.type === 'price_update') {
          setPrices(prev => ({
            ...prev,
            [data.coin]: data.data
          }));
        }
      };

      ws.onclose = () => {
        console.log('❌ WebSocket disconnected');
        setWsConnected(false);
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // ============================================================
  // FETCH SIGNALS (on demand, not on every price tick)
  // ============================================================
  const fetchSignals = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get<ScanResponse>(`${API_BASE}/scan`);
      setSignals(response.data.signals);
      setBtcContext(response.data.market_context.btc);
      setMarketSummary(response.data.market_summary);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching signals:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchSignals();
  }, [fetchSignals]);

  // ============================================================
  // FETCH ANALYSIS (on user click)
  // ============================================================
  const fetchAnalysis = async (coin: string) => {
    setLoading(true);
    setSelectedCoin(coin);

    try {
      const params = new URLSearchParams({
        capital: capital.toString(),
        trade_type: tradeType,
        experience: experience,
        recent_losses: recentLosses.toString(),
        trades_today: tradesToday.toString()
      });

      if (reason) {
        params.append('reason', reason);
      }

      const response = await axios.get<AnalysisResult>(
        `${API_BASE}/analyze/${coin}?${params}`
      );
      setAnalysis(response.data);
    } catch (error) {
      console.error('Error fetching analysis:', error);
    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // HELPER FUNCTIONS
  // ============================================================
  const formatPrice = (price: number): string => {
    if (price < 0.001) return `$${price.toFixed(8)}`;
    if (price < 1) return `$${price.toFixed(6)}`;
    if (price < 1000) return `$${price.toFixed(2)}`;
    return `$${price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  };

  const getVerdictColor = (verdict: string): string => {
    switch (verdict) {
      case 'BUY': return '#10b981';
      case 'WAIT': return '#f59e0b';
      case 'AVOID': return '#ef4444';
      case 'BLOCKED': return '#7c3aed';
      default: return '#6b7280';
    }
  };

  const getTrendIcon = (trend: string): string => {
    switch (trend) {
      case 'UP': return '📈';
      case 'DOWN': return '📉';
      default: return '➡️';
    }
  };

  // ============================================================
  // RENDER
  // ============================================================
  return (
    <div className="dashboard">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <h1>🤖 Crypto AI Trading</h1>
          <span className="version">v4.1</span>
        </div>
        <div className="header-right">
          <span className={`ws-status ${wsConnected ? 'connected' : 'disconnected'}`}>
            {wsConnected ? '🟢 Live' : '🔴 Disconnected'}
          </span>
          <button onClick={fetchSignals} className="refresh-btn" disabled={loading}>
            {loading ? '⏳' : '🔄'} Refresh Analysis
          </button>
          {lastUpdate && (
            <span className="last-update">
              Last: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
        </div>
      </header>

      {/* Market Summary Banner */}
      <div className={`market-banner ${marketSummary.includes('🟢') ? 'positive' : marketSummary.includes('🔴') ? 'negative' : 'neutral'}`}>
        <span>{marketSummary || '⏳ Loading market data...'}</span>
      </div>

      {/* Main Content */}
      <div className="content">
        {/* Left Panel: BTC Context + User Inputs */}
        <aside className="left-panel">
          {/* BTC Context */}
          {btcContext && (
            <div className="btc-context">
              <h3>₿ BTC Context</h3>
              <div className="btc-price">
                {formatPrice(prices['BTC_USDT']?.price || btcContext.price)}
                <span className={btcContext.change_24h >= 0 ? 'up' : 'down'}>
                  {btcContext.change_24h >= 0 ? '+' : ''}{btcContext.change_24h}% (24h)
                </span>
              </div>
              <div className="trends">
                <div className="trend-row">
                  <span>1H</span>
                  <span className={btcContext.trend_1h.toLowerCase()}>
                    {getTrendIcon(btcContext.trend_1h)} {btcContext.trend_1h}
                  </span>
                </div>
                <div className="trend-row">
                  <span>4H</span>
                  <span className={btcContext.trend_4h.toLowerCase()}>
                    {getTrendIcon(btcContext.trend_4h)} {btcContext.trend_4h}
                  </span>
                </div>
                <div className="trend-row">
                  <span>1D</span>
                  <span className={btcContext.trend_1d.toLowerCase()}>
                    {getTrendIcon(btcContext.trend_1d)} {btcContext.trend_1d}
                  </span>
                </div>
              </div>
              <div className={`overall-trend ${btcContext.overall_trend.toLowerCase()}`}>
                Overall: <strong>{btcContext.overall_trend}</strong>
              </div>
              <div className={`alt-support ${btcContext.support_alts ? 'yes' : 'no'}`}>
                {btcContext.support_alts ? '✅ Supports Alt Trades' : '⚠️ Alts May Underperform'}
              </div>
            </div>
          )}

          {/* User Inputs */}
          <div className="user-inputs">
            <h3>⚙️ Your Settings</h3>

            <div className="input-group">
              <label>Capital ($)</label>
              <input
                type="number"
                value={capital}
                onChange={(e) => setCapital(Number(e.target.value))}
                min={100}
                step={100}
              />
            </div>

            <div className="input-group">
              <label>Trade Type</label>
              <select value={tradeType} onChange={(e) => setTradeType(e.target.value)}>
                <option value="SCALP">Scalp (minutes)</option>
                <option value="SHORT_TERM">Short Term (1-2 days)</option>
                <option value="SWING">Swing (2-7 days)</option>
                <option value="INVESTMENT">Investment (weeks+)</option>
              </select>
            </div>

            <div className="input-group">
              <label>Experience</label>
              <select value={experience} onChange={(e) => setExperience(e.target.value)}>
                <option value="BEGINNER">Beginner</option>
                <option value="INTERMEDIATE">Intermediate</option>
                <option value="ADVANCED">Advanced</option>
              </select>
            </div>

            <div className="input-group">
              <label>Reason for Trade</label>
              <select value={reason} onChange={(e) => setReason(e.target.value)}>
                <option value="">None</option>
                <option value="STRATEGY">Strategy</option>
                <option value="FOMO">FOMO</option>
                <option value="NEWS">News</option>
                <option value="TIP">Tip</option>
                <option value="DIP_BUY">Dip Buy</option>
              </select>
            </div>

            <div className="input-group">
              <label>Recent Losses</label>
              <input
                type="number"
                value={recentLosses}
                onChange={(e) => setRecentLosses(Number(e.target.value))}
                min={0}
                max={10}
              />
            </div>

            <div className="input-group">
              <label>Trades Today</label>
              <input
                type="number"
                value={tradesToday}
                onChange={(e) => setTradesToday(Number(e.target.value))}
                min={0}
                max={20}
              />
            </div>
          </div>
        </aside>

        {/* Center: Signal Cards */}
        <main className="main-panel">
          <h3>📊 Live Prices & Signals</h3>

          <div className="signals-grid">
            {['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT'].map(coin => {
              const priceData = prices[coin];
              const signal = signals.find(s => s.coin === coin);

              return (
                <div
                  key={coin}
                  className={`signal-card ${selectedCoin === coin ? 'selected' : ''} ${priceData?.direction || ''}`}
                  onClick={() => fetchAnalysis(coin)}
                >
                  {/* Card Header */}
                  <div className="card-header">
                    <span className="coin-name">{coin.replace('_', '/')}</span>
                    {signal && (
                      <span
                        className="verdict-badge"
                        style={{ backgroundColor: getVerdictColor(signal.verdict) }}
                      >
                        {signal.verdict}
                      </span>
                    )}
                  </div>

                  {/* Price */}
                  <div className={`price-display ${priceData?.direction || ''}`}>
                    <span className="price">
                      {priceData ? formatPrice(priceData.price) : '—'}
                    </span>
                    <span className="price-source">WEBSOCKET</span>
                  </div>

                  {/* 24h Change */}
                  {priceData && (
                    <div className={`change-24h ${priceData.change_24h >= 0 ? 'up' : 'down'}`}>
                      {priceData.change_24h >= 0 ? '+' : ''}{priceData.change_24h}%
                    </div>
                  )}

                  {/* Probabilities */}
                  {signal && signal.model_ran && (
                    <div className="probs">
                      <div className="prob-row">
                        <span>WIN</span>
                        <div className="prob-bar">
                          <div
                            className="prob-fill win"
                            style={{ width: `${signal.win_probability || 0}%` }}
                          />
                        </div>
                        <span>{signal.win_probability?.toFixed(1)}%</span>
                      </div>
                      <div className="prob-row">
                        <span>LOSS</span>
                        <div className="prob-bar">
                          <div
                            className="prob-fill loss"
                            style={{ width: `${signal.loss_probability || 0}%` }}
                          />
                        </div>
                        <span>{signal.loss_probability?.toFixed(1)}%</span>
                      </div>
                    </div>
                  )}

                  {/* Expectancy (NEW!) */}
                  {signal && signal.expectancy !== null && (
                    <div className={`expectancy ${signal.expectancy >= 0 ? 'positive' : 'negative'}`}>
                      <span>Expectancy:</span>
                      <span>{signal.expectancy >= 0 ? '+' : ''}{signal.expectancy?.toFixed(1)}%</span>
                    </div>
                  )}

                  {/* Scenario Count */}
                  {signal && signal.scenario_count > 0 && (
                    <div className="scenario-badge">
                      ⚠️ {signal.scenario_count} scenario(s) active
                    </div>
                  )}

                  <div className="card-footer">
                    Click for full analysis
                  </div>
                </div>
              );
            })}
          </div>
        </main>

        {/* Right Panel: Analysis Detail */}
        <aside className="right-panel">
          {loading ? (
            <div className="loading-panel">⏳ Loading analysis...</div>
          ) : analysis ? (
            <div className="analysis-detail">
              <h3>📋 {analysis.coin.replace('_', '/')} Analysis</h3>

              {/* Capital Display */}
              <div className="capital-display">
                Capital: <strong>${analysis.capital}</strong>
              </div>

              {/* Verdict Box */}
              <div
                className="verdict-box"
                style={{ backgroundColor: getVerdictColor(analysis.verdict) + '20', borderColor: getVerdictColor(analysis.verdict) }}
              >
                <span className="verdict-text" style={{ color: getVerdictColor(analysis.verdict) }}>
                  {analysis.verdict}
                </span>
                <span className="confidence-text">{analysis.confidence} confidence</span>
              </div>

              {/* Probabilities */}
              {analysis.model_ran && (
                <div className="probs-section">
                  <div className="prob-display">
                    <div className="prob-item">
                      <span className="prob-label">WIN</span>
                      <div className="prob-bar-large">
                        <div
                          className="prob-fill win"
                          style={{ width: `${analysis.win_probability || 0}%` }}
                        />
                      </div>
                      <span className="prob-value">{analysis.win_probability}%</span>
                    </div>
                    <div className="prob-item">
                      <span className="prob-label">LOSS</span>
                      <div className="prob-bar-large">
                        <div
                          className="prob-fill loss"
                          style={{ width: `${analysis.loss_probability || 0}%` }}
                        />
                      </div>
                      <span className="prob-value">{analysis.loss_probability}%</span>
                    </div>
                  </div>
                </div>
              )}

              {/* NEW: Expectancy & Readiness */}
              {analysis.model_ran && (
                <div className="metrics-section">
                  <div className={`metric-card ${(analysis.expectancy || 0) >= 0 ? 'positive' : 'negative'}`}>
                    <span className="metric-label">Expectancy</span>
                    <span className="metric-value">
                      {(analysis.expectancy || 0) >= 0 ? '+' : ''}{analysis.expectancy}%
                    </span>
                    <span className="metric-status">{analysis.expectancy_status}</span>
                  </div>
                  <div className={`metric-card ${(analysis.readiness || 0) >= 0 ? 'positive' : 'negative'}`}>
                    <span className="metric-label">Readiness</span>
                    <span className="metric-value">
                      {(analysis.readiness || 0) >= 0 ? '+' : ''}{analysis.readiness}%
                    </span>
                    <span className="metric-status">{analysis.readiness_status}</span>
                  </div>
                </div>
              )}

              {/* Active Scenarios */}
              {analysis.active_scenarios.length > 0 && (
                <div className="scenarios-section">
                  <h4>⚠️ Active Scenarios</h4>
                  {analysis.active_scenarios.map((s, i) => (
                    <div key={i} className={`scenario-card ${s.severity.toLowerCase()}`}>
                      <span className="scenario-icon">{s.icon}</span>
                      <div className="scenario-content">
                        <div className="scenario-title">{s.title}</div>
                        <div className="scenario-message">{s.message}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Reasoning */}
              <div className="reasoning-section">
                <h4>💭 Reasoning</h4>
                {analysis.reasoning.map((r, i) => (
                  <div key={i} className="reasoning-item">{r}</div>
                ))}
              </div>

              {/* Warnings */}
              {analysis.warnings.length > 0 && (
                <div className="warnings-section">
                  <h4>⚠️ Warnings</h4>
                  {analysis.warnings.map((w, i) => (
                    <div key={i} className="warning-item">{w}</div>
                  ))}
                </div>
              )}

              {/* Forecast */}
              <div className="forecast-section">
                <h4>🔮 Forecast ({analysis.forecast.direction})</h4>
                <div className="targets">
                  <div className="target bull">
                    <span>Bull Target</span>
                    <span>{formatPrice(analysis.forecast.bull_target)}</span>
                  </div>
                  <div className="target current">
                    <span>Current</span>
                    <span>{formatPrice(analysis.forecast.current_price)}</span>
                  </div>
                  <div className="target bear">
                    <span>Bear Target</span>
                    <span>{formatPrice(analysis.forecast.bear_target)}</span>
                  </div>
                </div>
              </div>

              {/* Risk Management */}
              {analysis.risk.action === 'OPEN_POSITION' && (
                <div className="risk-section">
                  <h4>⚖️ Risk Management</h4>
                  <div className="risk-grid">
                    <div className="risk-item">
                      <span>Position Size</span>
                      <span>${analysis.risk.position_size_usd} ({analysis.risk.position_size_pct}%)</span>
                    </div>
                    <div className="risk-item">
                      <span>Entry</span>
                      <span>{formatPrice(analysis.risk.entry_price || 0)}</span>
                    </div>
                    <div className="risk-item loss">
                      <span>Stop Loss</span>
                      <span>{formatPrice(analysis.risk.stop_loss_price || 0)} (-{analysis.risk.stop_loss_pct}%)</span>
                    </div>
                    <div className="risk-item win">
                      <span>Take Profit</span>
                      <span>{formatPrice(analysis.risk.take_profit_price || 0)} (+{analysis.risk.take_profit_pct}%)</span>
                    </div>
                    <div className="risk-item loss">
                      <span>Max Loss</span>
                      <span>${analysis.risk.max_loss_usd}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Suggested Action */}
              <div className="action-section">
                <h4>🎯 Suggested Action</h4>
                <div className="action-box">
                  <div className="action-name">{analysis.suggested_action.action}</div>
                  <div className="action-message">{analysis.suggested_action.message}</div>
                  {analysis.suggested_action.next_check && (
                    <div className="next-check">Next check: {analysis.suggested_action.next_check}</div>
                  )}
                  {analysis.suggested_action.conditions && (
                    <div className="conditions">
                      {analysis.suggested_action.conditions.filter(c => c).map((c, i) => (
                        <div key={i} className="condition-item">• {c}</div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Market Regime */}
              <div className="regime-section">
                <h4>📊 Market Regime</h4>
                <div className="regime-info">
                  <span className={`regime-badge ${analysis.market_context.regime.regime.toLowerCase()}`}>
                    {analysis.market_context.regime.regime}
                  </span>
                  <span>ADX: {analysis.market_context.regime.adx}</span>
                  <span>Volatility: {analysis.market_context.regime.volatility}</span>
                </div>
                {analysis.market_context.regime.recommendation && (
                  <div className="regime-recommendation">
                    {analysis.market_context.regime.recommendation}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="no-selection">
              <div className="no-selection-icon">👆</div>
              <p>Click a coin card to see full analysis</p>
              <p className="hint">Analysis will use your settings from the left panel</p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
};

export default Dashboard;