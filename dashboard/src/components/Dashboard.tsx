import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Dashboard.css';
import NewsFeed from './NewsFeed';
import { useAuth } from '../context/AuthContext';
import { PortfolioPnL, HoldingSnapshot } from '../lib/supabase';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_URL = process.env.REACT_APP_WS_URL || (
  API_BASE ? API_BASE.replace(/^http/, 'ws') + '/ws/prices' : `ws://${window.location.host}/ws/prices`
);

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


interface PortfolioHoldingWithPrice {
  coin: string;
  amount: number;
  avgPrice: number;
  totalInvested: number;
  currentPrice: number;
}

type TimePeriod = 1 | 7 | 30;

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

const AVAILABLE_COINS = Object.keys(COIN_CONFIG);

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const { user, profile, signOut, holdings, recordSnapshot, getPortfolioPnL, addHolding, reduceHolding, updateCapital, refreshHoldings } = useAuth();

  const [prices, setPrices] = useState<Record<string, PriceData>>({});
  const [signals, setSignals] = useState<Signal[]>([]);
  const [marketSummary, setMarketSummary] = useState<string>('');
  const [wsConnected, setWsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [initialDataLoaded, setInitialDataLoaded] = useState(false);

  // Portfolio with current prices
  const [portfolioWithPrices, setPortfolioWithPrices] = useState<PortfolioHoldingWithPrice[]>([]);

  // Time period for P&L tracking
  const [timePeriod, setTimePeriod] = useState<TimePeriod>(7);
  const [periodPnL, setPeriodPnL] = useState<PortfolioPnL | null>(null);
  const [snapshotRecorded, setSnapshotRecorded] = useState(false);

  // Add Holding Modal State
  const [showAddModal, setShowAddModal] = useState(false);
  const [addForm, setAddForm] = useState({ coin: 'BTC_USDT', capital: '', price: '' });
  const [addLoading, setAddLoading] = useState(false);

  // Edit Capital Modal State
  const [showCapitalModal, setShowCapitalModal] = useState(false);
  const [capitalInput, setCapitalInput] = useState('');

  // USD to INR exchange rate
  const [usdToInr, setUsdToInr] = useState<number | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const priceBufferRef = useRef<Record<string, PriceData>>({});
  const updateTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const holdingsUpdateTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastHoldingsRef = useRef<string>('');

  // ============================================================
  // WEBSOCKET (throttled updates to prevent jitter)
  // ============================================================
  useEffect(() => {
    const flushPriceUpdates = () => {
      if (Object.keys(priceBufferRef.current).length > 0) {
        setPrices(prev => {
          const newPrices = { ...prev, ...priceBufferRef.current };
          updatePortfolioPrices(newPrices);
          return newPrices;
        });
        priceBufferRef.current = {};
      }
    };

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
          updatePortfolioPrices(data.prices);
          setInitialDataLoaded(true);
        } else if (data.type === 'price_update') {
          // Buffer updates and flush every 500ms to prevent jitter
          priceBufferRef.current[data.coin] = data.data;
          if (!updateTimeoutRef.current) {
            updateTimeoutRef.current = setTimeout(() => {
              flushPriceUpdates();
              updateTimeoutRef.current = null;
            }, 500);
          }
        }
      };

      ws.onclose = () => {
        console.log('❌ WebSocket disconnected');
        setWsConnected(false);
        setTimeout(connectWebSocket, 3000);
      };

      wsRef.current = ws;
    };

    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (updateTimeoutRef.current) clearTimeout(updateTimeoutRef.current);
    };
  }, []);

  const updatePortfolioPrices = useCallback((currentPrices: Record<string, PriceData>) => {
    if (holdings.length === 0) {
      setPortfolioWithPrices([]);  // Clear when no holdings
      return;
    }

    const updated: PortfolioHoldingWithPrice[] = holdings.map(h => ({
      coin: h.coin,
      amount: h.amount,
      avgPrice: h.avg_price,
      totalInvested: h.total_invested,
      currentPrice: currentPrices[h.coin]?.price || 0
    }));

    setPortfolioWithPrices(updated);
  }, [holdings]);

  // ============================================================
  // FETCH SIGNALS
  // ============================================================
  const fetchSignals = useCallback(async () => {
    try {
      const response = await axios.get<ScanResponse>(`${API_BASE}/scan`);
      setSignals(response.data.signals);
      setMarketSummary(response.data.market_summary);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching signals:', error);
    }
  }, []);

  useEffect(() => {
    fetchSignals();
    const interval = setInterval(fetchSignals, 60000);
    return () => clearInterval(interval);
  }, [fetchSignals]);

  // Fetch USD to INR exchange rate
  useEffect(() => {
    const fetchUsdToInr = async () => {
      try {
        const response = await axios.get('https://api.exchangerate-api.com/v4/latest/USD');
        setUsdToInr(response.data.rates.INR);
      } catch (error) {
        console.error('Error fetching USD/INR rate:', error);
      }
    };

    fetchUsdToInr();
    // Refresh every 10 minutes
    const interval = setInterval(fetchUsdToInr, 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

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

  const formatLargeNumber = (num: number): string => {
    if (num >= 1000000) return `$${(num / 1000000).toFixed(2)}M`;
    if (num >= 1000) return `$${(num / 1000).toFixed(2)}K`;
    if (num >= 1) return `$${num.toFixed(2)}`;
    if (num >= 0.01) return `$${num.toFixed(2)}`;
    if (num > 0) return `$${num.toFixed(4)}`; // Handle very small values like PEPE
    return `$0.00`;
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

  // Handle adding a new holding
  const handleAddHolding = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!addForm.capital || !addForm.price) return;

    const capital = parseFloat(addForm.capital);
    const price = parseFloat(addForm.price);
    const coinAmount = capital / price; // Calculate how many coins for the capital

    setAddLoading(true);
    const success = await addHolding(
      addForm.coin,
      coinAmount,
      price
    );

    if (success) {
      setShowAddModal(false);
      setAddForm({ coin: 'BTC_USDT', capital: '', price: '' });
      setSnapshotRecorded(false); // Re-record snapshot with new holdings
    }
    setAddLoading(false);
  };

  // Calculate preview of coins to receive
  const getCoinsPreview = (): string => {
    if (!addForm.capital || !addForm.price) return '';
    const capital = parseFloat(addForm.capital);
    const price = parseFloat(addForm.price);
    if (price <= 0) return '';
    const coins = capital / price;
    if (coins >= 1000000000) return `${(coins / 10000000).toFixed(2)} Cr`;
    if (coins >= 10000000) return `${(coins / 10000000).toFixed(2)} Cr`;
    if (coins >= 100000) return `${(coins / 100000).toFixed(2)} L`;
    if (coins >= 1000) return `${(coins / 1000).toFixed(2)}K`;
    if (coins >= 1) return coins.toFixed(4);
    return coins.toFixed(8);
  };

  // Handle selling/removing holding
  const handleSellHolding = async (coin: string, amount: number) => {
    const coinName = coin.split('_')[0];
    if (window.confirm(`Sell all ${coinName}?`)) {
      const success = await reduceHolding(coin, amount);
      if (success) {
        await refreshHoldings();
        setSnapshotRecorded(false);
      }
    }
  };

  // Handle updating capital
  const handleUpdateCapital = async (e: React.FormEvent) => {
    e.preventDefault();
    const newCapital = parseFloat(capitalInput);
    if (isNaN(newCapital) || newCapital < 0) return;

    const success = await updateCapital(newCapital);
    if (success) {
      setShowCapitalModal(false);
      setCapitalInput('');
    }
  };

  // Use current price for add form
  const useCurrentPrice = () => {
    const currentPrice = prices[addForm.coin]?.price;
    if (currentPrice) {
      setAddForm(prev => ({ ...prev, price: currentPrice.toString() }));
    }
  };

  // Portfolio calculations
  const portfolioValue = portfolioWithPrices.reduce((sum, h) => sum + (h.amount * h.currentPrice), 0);
  const portfolioCost = portfolioWithPrices.reduce((sum, h) => sum + h.totalInvested, 0);
  const portfolioPnL = portfolioValue - portfolioCost;
  const portfolioPnLPercent = portfolioCost > 0 ? (portfolioPnL / portfolioCost) * 100 : 0;

  // Record snapshot and fetch P&L when portfolio value changes
  useEffect(() => {
    // Record snapshot if we have holdings (even if value is tiny)
    const hasHoldings = portfolioWithPrices.length > 0 && portfolioWithPrices.some(h => h.currentPrice > 0);

    if (hasHoldings && !snapshotRecorded && user) {
      const snapshot: HoldingSnapshot[] = portfolioWithPrices.map(h => ({
        coin: h.coin,
        amount: h.amount,
        price: h.currentPrice,
        value: h.amount * h.currentPrice
      }));

      console.log('Recording snapshot:', { portfolioValue, snapshot });
      recordSnapshot(portfolioValue, snapshot).then((success) => {
        console.log('Snapshot recorded:', success);
        setSnapshotRecorded(true);
      });
    }
  }, [portfolioValue, snapshotRecorded, user, portfolioWithPrices, recordSnapshot]);

  // Fetch P&L for selected time period
  useEffect(() => {
    if (user && snapshotRecorded) {
      console.log('Fetching P&L for period:', timePeriod);
      getPortfolioPnL(timePeriod).then(pnl => {
        console.log('P&L result:', pnl);
        setPeriodPnL(pnl);
      });
    }
  }, [user, timePeriod, snapshotRecorded, getPortfolioPnL]);

  // Update portfolio with prices when holdings or prices change (debounced)
  useEffect(() => {
    if (Object.keys(prices).length === 0) return;

    // Create a key to detect actual holdings changes
    const holdingsKey = holdings.map(h => `${h.coin}:${h.amount}`).join(',');

    // If holdings changed, debounce the update to prevent jitter
    if (holdingsKey !== lastHoldingsRef.current) {
      lastHoldingsRef.current = holdingsKey;

      // Clear any pending update
      if (holdingsUpdateTimeoutRef.current) {
        clearTimeout(holdingsUpdateTimeoutRef.current);
      }

      // Debounce holdings update by 100ms
      holdingsUpdateTimeoutRef.current = setTimeout(() => {
        updatePortfolioPrices(prices);
        holdingsUpdateTimeoutRef.current = null;
      }, 100);
    } else {
      // Price-only updates can happen immediately
      updatePortfolioPrices(prices);
    }

    return () => {
      if (holdingsUpdateTimeoutRef.current) {
        clearTimeout(holdingsUpdateTimeoutRef.current);
      }
    };
  }, [holdings, prices, updatePortfolioPrices]);

  // ============================================================
  // RENDER
  // ============================================================

  // Show loading screen until WebSocket connects and initial data is loaded
  if (!initialDataLoaded) {
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
            <span>{wsConnected ? 'Connected, loading prices...' : 'Establishing connection...'}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <h1>🤖 TradeWise Trading</h1>
          <span className="version">v5.0</span>
        </div>
        <div className="header-right">
          <span className={`ws-status ${wsConnected ? 'connected' : 'disconnected'}`}>
            {wsConnected ? '🟢 Live' : '🔴 Offline'}
          </span>
          <button onClick={() => navigate('/backtest')} className="refresh-btn">Backtest</button>
          <button onClick={() => navigate('/signals')} className="refresh-btn">Signals</button>
          <button onClick={() => navigate('/paper-trading')} className="refresh-btn">Paper Trading</button>
          <button onClick={fetchSignals} className="refresh-btn">Refresh</button>
          {lastUpdate && <span className="last-update">{lastUpdate.toLocaleTimeString()}</span>}

          {/* USD to INR Rate */}
          {usdToInr && (
            <div className="usd-inr-rate">
              <span className="rate-label">USD/INR</span>
              <span className="rate-value">₹{usdToInr.toFixed(2)}</span>
            </div>
          )}

          {/* User Profile */}
          <div className="user-menu">
            <span className="user-avatar">
              {profile?.avatar_url ? (
                <img src={profile.avatar_url} alt="avatar" />
              ) : (
                <span className="avatar-placeholder">
                  {(profile?.username || user?.email || 'U').charAt(0).toUpperCase()}
                </span>
              )}
            </span>
            <span className="user-name">{profile?.display_name || profile?.username || user?.email}</span>
            <button onClick={signOut} className="logout-btn">Logout</button>
          </div>
        </div>
      </header>

      {/* Market Banner */}
      <div className={`market-banner ${marketSummary.includes('🟢') ? 'positive' : marketSummary.includes('🔴') ? 'negative' : 'neutral'}`}>
        {marketSummary || '⏳ Loading market data...'}
      </div>

      {/* Main Content - 2 Column */}
      <div className="content">
        {/* LEFT: Portfolio + News */}
        <aside className="left-panel">
          {/* Portfolio */}
          <div className="card portfolio-card">
            <div className="portfolio-header">
              <h3>💰 My Portfolio</h3>
              <div className="portfolio-actions">
                <div className="time-period-selector">
                  {([1, 7, 30] as TimePeriod[]).map(period => (
                    <button
                      key={period}
                      className={`period-btn ${timePeriod === period ? 'active' : ''}`}
                      onClick={() => setTimePeriod(period)}
                    >
                      {period}D
                    </button>
                  ))}
                </div>
                <button className="add-holding-btn" onClick={() => setShowAddModal(true)}>
                  + Add
                </button>
              </div>
            </div>

            <div className="portfolio-total">
              <span className="portfolio-label">Total Value</span>
              <span className="portfolio-value">{formatLargeNumber(portfolioValue)}</span>
              <span className={`portfolio-pnl ${portfolioPnL >= 0 ? 'positive' : 'negative'}`}>
                {portfolioPnL >= 0 ? '+' : ''}{formatLargeNumber(portfolioPnL)} ({portfolioPnLPercent.toFixed(2)}%)
              </span>
            </div>

            {/* Period P&L */}
            {portfolioWithPrices.length > 0 && (
              <div className={`period-pnl ${periodPnL ? (periodPnL.pnl_amount >= 0 ? 'positive' : 'negative') : 'neutral'}`}>
                <span className="period-label">{timePeriod}D Change</span>
                <span className="period-value">
                  {periodPnL ? (
                    <>
                      {periodPnL.pnl_amount >= 0 ? '+' : ''}{formatLargeNumber(periodPnL.pnl_amount)}
                      <span className="period-percent">
                        ({periodPnL.pnl_percent >= 0 ? '+' : ''}{periodPnL.pnl_percent.toFixed(2)}%)
                      </span>
                    </>
                  ) : (
                    <span className="period-no-data">Tracking started</span>
                  )}
                </span>
              </div>
            )}

            {/* Capital */}
            <div className="capital-row">
              <span className="capital-label">Available Capital</span>
              <span className="capital-value">{formatLargeNumber(profile?.capital || 0)}</span>
              <button
                className="edit-capital-btn"
                onClick={() => {
                  setCapitalInput((profile?.capital || 0).toString());
                  setShowCapitalModal(true);
                }}
                title="Edit capital"
              >
                ✏️
              </button>
            </div>

            <div className="portfolio-holdings">
              {portfolioWithPrices.length === 0 ? (
                <div className="no-holdings">
                  <p>No holdings yet</p>
                  <p className="no-holdings-hint">Add coins to start tracking</p>
                </div>
              ) : (
                portfolioWithPrices.map(h => {
                  const value = h.amount * h.currentPrice;
                  const pnlPct = h.avgPrice > 0 ? ((h.currentPrice - h.avgPrice) / h.avgPrice) * 100 : 0;
                  const config = COIN_CONFIG[h.coin];
                  return (
                    <div key={h.coin} className="holding-row">
                      <div className="holding-coin">
                        <span className="holding-icon">{config?.icon}</span>
                        <div className="holding-details">
                          <span className="holding-name">{config?.name || h.coin}</span>
                          <span className="holding-amount-coin">
                            {h.amount < 1 ? h.amount.toFixed(6) : h.amount.toLocaleString()} {h.coin.split('_')[0]}
                          </span>
                        </div>
                      </div>
                      <div className="holding-value">
                        <span className="holding-amount">{formatLargeNumber(value)}</span>
                        <span className={`holding-pnl ${pnlPct >= 0 ? 'positive' : 'negative'}`}>
                          {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%
                        </span>
                      </div>
                      <button
                        className="sell-btn"
                        onClick={() => handleSellHolding(h.coin, h.amount)}
                        title="Sell all"
                      >
                        ×
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </div>

        </aside>

        {/* CENTER: Signal Cards (Full Width) */}
        <main className="main-panel">
          <h3>📊 Live Prices & Signals</h3>
          <p className="panel-subtitle">Click a coin for detailed analysis</p>

          <div className="signals-grid">
            {['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT'].map(coin => {
              const priceData = prices[coin];
              const signal = signals.find(s => s.coin === coin);
              const config = COIN_CONFIG[coin];

              // Mock time diffs (approximate)
              const change1h = ((priceData?.change_24h || 0) / 24).toFixed(2);
              const change7d = ((priceData?.change_24h || 0) * 3.5).toFixed(2);

              return (
                <div
                  key={coin}
                  className={`signal-card ${priceData?.direction || ''}`}
                  onClick={() => navigate(`/coin/${coin}`)}
                >
                  {/* Header */}
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

                  {/* Price */}
                  <div className={`price-section ${priceData?.direction || ''}`}>
                    <span className="price">{priceData ? formatPrice(priceData.price) : '—'}</span>
                  </div>

                  {/* Time Diffs */}
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

                  {/* Expectancy */}
                  {signal && signal.expectancy !== null && (
                    <div className={`expectancy-row ${signal.expectancy >= 0 ? 'positive' : 'negative'}`}>
                      <span>Expectancy</span>
                      <span>{signal.expectancy >= 0 ? '+' : ''}{signal.expectancy.toFixed(1)}%</span>
                    </div>
                  )}

                  {/* Scenario Badge */}
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

          {/* News Feed - Arkham-style cards */}
          <NewsFeed />
        </main>
      </div>

      {/* Add Holding Modal */}
      {showAddModal && (
        <div className="modal-overlay" onClick={() => setShowAddModal(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Add Holding</h3>
              <button className="modal-close" onClick={() => setShowAddModal(false)}>×</button>
            </div>
            <form onSubmit={handleAddHolding}>
              <div className="form-group">
                <label>Coin</label>
                <select
                  value={addForm.coin}
                  onChange={e => setAddForm(prev => ({ ...prev, coin: e.target.value }))}
                >
                  {AVAILABLE_COINS.map(coin => (
                    <option key={coin} value={coin}>
                      {COIN_CONFIG[coin].icon} {COIN_CONFIG[coin].name} ({coin.replace('_', '/')})
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label>Capital to Invest (USD)</label>
                <input
                  type="number"
                  step="any"
                  placeholder="e.g., 500"
                  value={addForm.capital}
                  onChange={e => setAddForm(prev => ({ ...prev, capital: e.target.value }))}
                  required
                />
              </div>
              <div className="form-group">
                <label>
                  Purchase Price per Coin (USD)
                  <button type="button" className="use-current-btn" onClick={useCurrentPrice}>
                    Use Current
                  </button>
                </label>
                <input
                  type="number"
                  step="any"
                  placeholder="e.g., 0.00000470"
                  value={addForm.price}
                  onChange={e => setAddForm(prev => ({ ...prev, price: e.target.value }))}
                  required
                />
              </div>
              {/* Coins Preview */}
              {getCoinsPreview() && (
                <div className="coins-preview">
                  <span className="preview-label">You'll receive:</span>
                  <span className="preview-value">
                    ~{getCoinsPreview()} {addForm.coin.split('_')[0]}
                  </span>
                </div>
              )}
              <div className="form-actions">
                <button type="button" className="cancel-btn" onClick={() => setShowAddModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="submit-btn" disabled={addLoading}>
                  {addLoading ? 'Adding...' : 'Add Holding'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit Capital Modal */}
      {showCapitalModal && (
        <div className="modal-overlay" onClick={() => setShowCapitalModal(false)}>
          <div className="modal modal-small" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Edit Capital</h3>
              <button className="modal-close" onClick={() => setShowCapitalModal(false)}>×</button>
            </div>
            <form onSubmit={handleUpdateCapital}>
              <div className="form-group">
                <label>Available Capital (USD)</label>
                <input
                  type="number"
                  step="any"
                  min="0"
                  placeholder="e.g., 10000"
                  value={capitalInput}
                  onChange={e => setCapitalInput(e.target.value)}
                  required
                  autoFocus
                />
              </div>
              <div className="form-actions">
                <button type="button" className="cancel-btn" onClick={() => setShowCapitalModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="submit-btn">
                  Update Capital
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;