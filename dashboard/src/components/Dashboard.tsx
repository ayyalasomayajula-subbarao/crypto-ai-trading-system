import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Dashboard.css';
import NewsFeed from './NewsFeed';

const API_BASE = process.env.REACT_APP_API_URL || window.location.origin;
const WS_URL = process.env.REACT_APP_WS_URL || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/prices`;

// ============================================================
// TYPES
// ============================================================

interface PriceData {
  price: number;
  change_24h: number;
  high_24h?: number;
  low_24h?: number;
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

interface ScanResponse {
  timestamp: string;
  market_context: { btc: BTCContext };
  signals: Signal[];
  market_summary: string;
}

// ============================================================
// CONFIG
// ============================================================

const COIN_CONFIG: Record<string, { icon: string; name: string; color: string }> = {
  'BTC_USDT': { icon: '₿', name: 'Bitcoin', color: '#f7931a' },
  'ETH_USDT': { icon: 'Ξ', name: 'Ethereum', color: '#627eea' },
  'SOL_USDT': { icon: '◎', name: 'Solana', color: '#00ffa3' },
  'PEPE_USDT': { icon: '🐸', name: 'Pepe', color: '#4a9c2d' }
};

// ============================================================
// MAIN COMPONENT
// ============================================================

const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  const [prices, setPrices] = useState<Record<string, PriceData>>({});
  const [signals, setSignals] = useState<Signal[]>([]);
  const [marketSummary, setMarketSummary] = useState<string>('');
  const [wsConnected, setWsConnected] = useState(false);
  const [initialDataLoaded, setInitialDataLoaded] = useState(false);
  const [signalsLoaded, setSignalsLoaded] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);

  // ============================================================
  // WEBSOCKET
  // ============================================================
  useEffect(() => {
    const connectWebSocket = () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setWsConnected(true);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'initial' || data.type === 'all_prices') {
          setPrices(data.prices);
          setInitialDataLoaded(true);
        } else if (data.type === 'price_update') {
          setPrices(prev => ({ ...prev, [data.coin]: data.data }));
        }
      };

      ws.onclose = () => {
        console.log('❌ WebSocket disconnected');
        setWsConnected(false);
        setTimeout(connectWebSocket, 3000);
      };

      wsRef.current = ws;
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        connectWebSocket();
      }
    };

    connectWebSocket();
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      if (wsRef.current) wsRef.current.close();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ============================================================
  // FETCH SIGNALS
  // ============================================================
  const fetchSignals = useCallback(async () => {
    try {
      const response = await axios.get<ScanResponse>(`${API_BASE}/scan`);
      setSignals(response.data.signals);
      setMarketSummary(response.data.market_summary);
      setSignalsLoaded(true);
    } catch (error) {
      console.error('Error fetching signals:', error);
      setSignalsLoaded(true);
    }
  }, []);

  useEffect(() => {
    fetchSignals();
    const interval = setInterval(fetchSignals, 60000);
    return () => clearInterval(interval);
  }, [fetchSignals]);

  // ============================================================
  // HELPERS
  // ============================================================
  const formatPrice = (price: number): string => {
    if (price < 0.0001) return `$${price.toFixed(8)}`;
    if (price < 0.01) return `$${price.toFixed(6)}`;
    if (price < 1) return `$${price.toFixed(4)}`;
    if (price < 1000) return `$${price.toFixed(2)}`;
    return `$${price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
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

  // ============================================================
  // RENDER
  // ============================================================

  if (!initialDataLoaded || !signalsLoaded) {
    return (
      <div className="dashboard loading-screen">
        <div className="loading-content">
          <div className="loading-logo">🤖</div>
          <h1 className="loading-title">TradeWise Trading</h1>
          <div className="loading-spinner-container">
            <div className="loading-spinner-ring"></div>
          </div>
          <p className="loading-text">Connecting to live markets...</p>
          <div className="loading-status">
            <span className={`status-dot ${wsConnected ? 'connected' : ''}`}></span>
            <span>{!wsConnected ? 'Establishing connection...' : !initialDataLoaded ? 'Loading prices...' : 'Loading market signals...'}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard no-header">
      {/* Market Banner */}
      <div className={`market-banner ${marketSummary.includes('🟢') ? 'positive' : marketSummary.includes('🔴') ? 'negative' : 'neutral'}`}>
        {marketSummary || '⏳ Loading market data...'}
      </div>

      {/* Main Content */}
      <div className="content content-full">
          <div className="signals-grid">
            {['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT'].map(coin => {
              const priceData = prices[coin];
              const signal = signals.find(s => s.coin === coin);
              const config = COIN_CONFIG[coin];

              const change1h = ((priceData?.change_24h || 0) / 24).toFixed(2);
              const change7d = ((priceData?.change_24h || 0) * 3.5).toFixed(2);

              return (
                <div
                  key={coin}
                  className={`signal-card ${priceData?.direction || ''}`}
                  onClick={() => navigate(`/coin/${coin}`)}
                >
                  <div className="card-header">
                    <div className="coin-info">
                      <span className="coin-icon" style={{ color: config.color }}>{config.icon}</span>
                      <div>
                        <span className="coin-symbol">{coin.replace('_', '/')}</span>
                        <span className="coin-name">{config.name}</span>
                      </div>
                    </div>
                    {signal && (
                      <span
                        className="verdict-badge"
                        style={{
                          backgroundColor: `${getVerdictColor(signal.verdict)}15`,
                          color: getVerdictColor(signal.verdict),
                          border: `1px solid ${getVerdictColor(signal.verdict)}30`
                        }}
                      >
                        {signal.verdict}
                      </span>
                    )}
                  </div>

                  <div className={`price-section ${priceData?.direction || ''}`}>
                    <span className="price">{priceData ? formatPrice(priceData.price) : '—'}</span>
                  </div>

                  <div className="time-diffs">
                    <div className="time-diff">
                      <span className="diff-label">1H</span>
                      <span className={`diff-value ${parseFloat(change1h) >= 0 ? 'positive' : 'negative'}`}>
                        {parseFloat(change1h) >= 0 ? '+' : ''}{change1h}%
                      </span>
                    </div>
                    <div className="time-diff">
                      <span className="diff-label">24H</span>
                      <span className={`diff-value ${(priceData?.change_24h || 0) >= 0 ? 'positive' : 'negative'}`}>
                        {(priceData?.change_24h || 0) >= 0 ? '+' : ''}{(priceData?.change_24h || 0).toFixed(2)}%
                      </span>
                    </div>
                    <div className="time-diff">
                      <span className="diff-label">7D</span>
                      <span className={`diff-value ${parseFloat(change7d) >= 0 ? 'positive' : 'negative'}`}>
                        {parseFloat(change7d) >= 0 ? '+' : ''}{change7d}%
                      </span>
                    </div>
                  </div>

                  {signal && signal.expectancy !== null && (
                    <div className={`expectancy-row ${signal.expectancy >= 0 ? 'positive' : 'negative'}`}>
                      <span>Expectancy</span>
                      <span>{signal.expectancy >= 0 ? '+' : ''}{signal.expectancy.toFixed(1)}%</span>
                    </div>
                  )}

                  {signal && signal.scenario_count > 0 && (
                    <div className="scenario-badge">
                      ⚠️ {signal.scenario_count} scenario{signal.scenario_count > 1 ? 's' : ''}
                    </div>
                  )}

                  <div className="card-footer">
                    Click for detailed analysis →
                  </div>
                </div>
              );
            })}
          </div>

          {/* News Feed */}
          <NewsFeed />
      </div>
    </div>
  );
};

export default Dashboard;
