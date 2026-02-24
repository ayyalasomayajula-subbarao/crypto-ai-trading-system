import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { PortfolioPnL, HoldingSnapshot } from '../lib/supabase';
import '../components/Dashboard.css';

const WS_URL = process.env.REACT_APP_WS_URL || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/prices`;

interface PriceData {
  price: number;
  change_24h: number;
  direction: 'up' | 'down' | 'same';
  timestamp: string;
}

interface PortfolioHoldingWithPrice {
  coin: string;
  amount: number;
  avgPrice: number;
  totalInvested: number;
  currentPrice: number;
}

type TimePeriod = 1 | 7 | 30;

const COIN_CONFIG: Record<string, { icon: string; name: string; color: string }> = {
  'BTC_USDT': { icon: '₿', name: 'Bitcoin', color: '#f7931a' },
  'ETH_USDT': { icon: 'Ξ', name: 'Ethereum', color: '#627eea' },
  'SOL_USDT': { icon: '◎', name: 'Solana', color: '#00ffa3' },
  'PEPE_USDT': { icon: '🐸', name: 'Pepe', color: '#4a9c2d' },
};

const AVAILABLE_COINS = Object.keys(COIN_CONFIG);

const PortfolioPage: React.FC = () => {
  const {
    user, profile, holdings,
    recordSnapshot, getPortfolioPnL,
    addHolding, reduceHolding, updateCapital, refreshHoldings,
  } = useAuth();

  const [prices, setPrices] = useState<Record<string, PriceData>>({});
  const [portfolioWithPrices, setPortfolioWithPrices] = useState<PortfolioHoldingWithPrice[]>([]);
  const [timePeriod, setTimePeriod] = useState<TimePeriod>(7);
  const [periodPnL, setPeriodPnL] = useState<PortfolioPnL | null>(null);
  const [snapshotRecorded, setSnapshotRecorded] = useState(false);

  // Add Holding Modal
  const [showAddModal, setShowAddModal] = useState(false);
  const [addForm, setAddForm] = useState({ coin: 'BTC_USDT', capital: '', price: '' });
  const [addLoading, setAddLoading] = useState(false);

  // Edit Capital Modal
  const [showCapitalModal, setShowCapitalModal] = useState(false);
  const [capitalInput, setCapitalInput] = useState('');

  const wsRef = useRef<WebSocket | null>(null);
  const holdingsUpdateTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastHoldingsRef = useRef<string>('');

  // ============================================================
  // WEBSOCKET
  // ============================================================
  useEffect(() => {
    const ws = new WebSocket(WS_URL);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'initial' || data.type === 'all_prices') {
        setPrices(data.prices);
      } else if (data.type === 'price_update') {
        setPrices(prev => ({ ...prev, [data.coin]: data.data }));
      }
    };

    ws.onclose = () => {
      setTimeout(() => {
        wsRef.current = new WebSocket(WS_URL);
      }, 3000);
    };

    wsRef.current = ws;
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, []);

  // ============================================================
  // UPDATE PORTFOLIO WITH PRICES
  // ============================================================
  const updatePortfolioPrices = useCallback((currentPrices: Record<string, PriceData>) => {
    if (holdings.length === 0) {
      setPortfolioWithPrices([]);
      return;
    }
    const updated: PortfolioHoldingWithPrice[] = holdings.map(h => ({
      coin: h.coin,
      amount: h.amount,
      avgPrice: h.avg_price,
      totalInvested: h.total_invested,
      currentPrice: currentPrices[h.coin]?.price || 0,
    }));
    setPortfolioWithPrices(updated);
  }, [holdings]);

  // Debounced holdings update
  useEffect(() => {
    if (Object.keys(prices).length === 0) return;
    const holdingsKey = holdings.map(h => `${h.coin}:${h.amount}`).join(',');
    if (holdingsKey !== lastHoldingsRef.current) {
      lastHoldingsRef.current = holdingsKey;
      if (holdingsUpdateTimeoutRef.current) clearTimeout(holdingsUpdateTimeoutRef.current);
      holdingsUpdateTimeoutRef.current = setTimeout(() => {
        updatePortfolioPrices(prices);
        holdingsUpdateTimeoutRef.current = null;
      }, 100);
    } else {
      updatePortfolioPrices(prices);
    }
    return () => {
      if (holdingsUpdateTimeoutRef.current) clearTimeout(holdingsUpdateTimeoutRef.current);
    };
  }, [holdings, prices, updatePortfolioPrices]);

  // ============================================================
  // HELPERS
  // ============================================================
  const formatLargeNumber = (num: number): string => {
    if (num >= 1000000) return `$${(num / 1000000).toFixed(2)}M`;
    if (num >= 1000) return `$${(num / 1000).toFixed(2)}K`;
    if (num >= 1) return `$${num.toFixed(2)}`;
    if (num >= 0.01) return `$${num.toFixed(2)}`;
    if (num > 0) return `$${num.toFixed(4)}`;
    return `$0.00`;
  };

  // Portfolio calculations
  const portfolioValue = portfolioWithPrices.reduce((sum, h) => sum + (h.amount * h.currentPrice), 0);
  const portfolioCost = portfolioWithPrices.reduce((sum, h) => sum + h.totalInvested, 0);
  const portfolioPnL = portfolioValue - portfolioCost;
  const portfolioPnLPercent = portfolioCost > 0 ? (portfolioPnL / portfolioCost) * 100 : 0;

  // Record snapshot
  useEffect(() => {
    const hasHoldings = portfolioWithPrices.length > 0 && portfolioWithPrices.some(h => h.currentPrice > 0);
    if (hasHoldings && !snapshotRecorded && user) {
      const snapshot: HoldingSnapshot[] = portfolioWithPrices.map(h => ({
        coin: h.coin,
        amount: h.amount,
        price: h.currentPrice,
        value: h.amount * h.currentPrice,
      }));
      recordSnapshot(portfolioValue, snapshot).then(() => setSnapshotRecorded(true));
    }
  }, [portfolioValue, snapshotRecorded, user, portfolioWithPrices, recordSnapshot]);

  // Fetch P&L for selected time period
  useEffect(() => {
    if (user && snapshotRecorded) {
      getPortfolioPnL(timePeriod).then(pnl => setPeriodPnL(pnl));
    }
  }, [user, timePeriod, snapshotRecorded, getPortfolioPnL]);

  // ============================================================
  // HANDLERS
  // ============================================================
  const handleAddHolding = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!addForm.capital || !addForm.price) return;
    const capital = parseFloat(addForm.capital);
    const price = parseFloat(addForm.price);
    const coinAmount = capital / price;
    setAddLoading(true);
    const success = await addHolding(addForm.coin, coinAmount, price);
    if (success) {
      setShowAddModal(false);
      setAddForm({ coin: 'BTC_USDT', capital: '', price: '' });
      setSnapshotRecorded(false);
    }
    setAddLoading(false);
  };

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

  const useCurrentPrice = () => {
    const currentPrice = prices[addForm.coin]?.price;
    if (currentPrice) {
      setAddForm(prev => ({ ...prev, price: currentPrice.toString() }));
    }
  };

  // ============================================================
  // RENDER
  // ============================================================
  return (
    <div className="dashboard">
      <div className="content content-full">
        <h2 style={{ color: '#e2e8f0', marginBottom: 8 }}>Portfolio</h2>

        {/* Portfolio Card */}
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

          {/* Holdings */}
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

export default PortfolioPage;
