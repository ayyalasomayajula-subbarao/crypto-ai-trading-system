import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import SignalCard from './SignalCard';
import MarketContext from './MarketContext';
import CoinDetail from './CoinDetail';

const API_BASE = 'http://localhost:8000';

interface Signal {
  coin: string;
  price: number;
  price_source: string;
  win_probability: number;
  loss_probability: number;
  verdict: string;
  confidence: string;
  signal_strength: string;
  warnings_count: number;
}

interface BTCContext {
  trend_1h: string;
  trend_4h: string;
  trend_1d: string;
  overall_trend: string;
  strength: number;
  support_alts: boolean;
  price: number;
  change_24h: number;
}

interface ScanResponse {
  timestamp: string;
  market_context: {
    btc: BTCContext;
    overall_risk: string;
  };
  signals: Signal[];
  buy_signals: Signal[];
  market_summary: string;
}

const Dashboard: React.FC = () => {
  const [data, setData] = useState<ScanResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCoin, setSelectedCoin] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get<ScanResponse>(`${API_BASE}/scan`);
      setData(response.data);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError('Failed to fetch data. Is the API running?');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchData, 60000); // Refresh every 60s
      return () => clearInterval(interval);
    }
  }, [fetchData, autoRefresh]);

  const getVerdictColor = (verdict: string): string => {
    switch (verdict) {
      case 'BUY': return '#10b981';
      case 'HOLD': return '#3b82f6';
      case 'WAIT': return '#f59e0b';
      case 'AVOID': return '#ef4444';
      case 'EXIT': return '#ef4444';
      case 'BLOCKED': return '#7c3aed';
      default: return '#6b7280';
    }
  };

  const getRiskColor = (risk: string): string => {
    switch (risk) {
      case 'HIGH': return '#ef4444';
      case 'NORMAL': return '#10b981';
      default: return '#f59e0b';
    }
  };

  if (error) {
    return (
      <div className="dashboard error-state">
        <div className="error-container">
          <h2>⚠️ Connection Error</h2>
          <p>{error}</p>
          <p>Make sure the API is running:</p>
          <code>python api_v2.py</code>
          <button onClick={fetchData} className="retry-btn">
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-left">
          <h1>🤖 Crypto AI Trading</h1>
          <span className="version">v2.2 Phase 2</span>
        </div>
        <div className="header-right">
          <div className="auto-refresh">
            <label>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              Auto-refresh
            </label>
          </div>
          <button onClick={fetchData} className="refresh-btn" disabled={loading}>
            {loading ? '⏳' : '🔄'} Refresh
          </button>
          {lastUpdate && (
            <span className="last-update">
              Last: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
        </div>
      </header>

      {/* Market Summary Banner */}
      {data && (
        <div 
          className="market-summary"
          style={{ 
            backgroundColor: data.market_context.overall_risk === 'HIGH' 
              ? '#fef2f2' 
              : '#f0fdf4',
            borderColor: data.market_context.overall_risk === 'HIGH'
              ? '#ef4444'
              : '#10b981'
          }}
        >
          <span className="summary-text">{data.market_summary}</span>
          <span 
            className="risk-badge"
            style={{ backgroundColor: getRiskColor(data.market_context.overall_risk) }}
          >
            {data.market_context.overall_risk} RISK
          </span>
        </div>
      )}

      {/* Main Content */}
      <div className="dashboard-content">
        {/* Left: Market Context */}
        <aside className="sidebar">
          {data && <MarketContext btc={data.market_context.btc} />}
        </aside>

        {/* Center: Signal Cards */}
        <main className="main-content">
          <h2>📊 Trading Signals</h2>
          
          {loading && !data ? (
            <div className="loading">Loading signals...</div>
          ) : (
            <div className="signals-grid">
              {data?.signals.map((signal) => (
                <SignalCard
                  key={signal.coin}
                  signal={signal}
                  onClick={() => setSelectedCoin(signal.coin)}
                  isSelected={selectedCoin === signal.coin}
                  verdictColor={getVerdictColor(signal.verdict)}
                />
              ))}
            </div>
          )}

          {/* Buy Signals Alert */}
          {data && data.buy_signals.length > 0 && (
            <div className="buy-alert">
              <h3>🚨 BUY SIGNALS DETECTED!</h3>
              {data.buy_signals.map((signal) => (
                <div key={signal.coin} className="buy-signal-item">
                  <strong>{signal.coin}</strong>: {(signal.win_probability * 100).toFixed(1)}% WIN probability
                </div>
              ))}
            </div>
          )}
        </main>

        {/* Right: Coin Detail */}
        <aside className="detail-panel">
          {selectedCoin ? (
            <CoinDetail coin={selectedCoin} />
          ) : (
            <div className="no-selection">
              <p>👆 Click a coin card to see details</p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
};

export default Dashboard;