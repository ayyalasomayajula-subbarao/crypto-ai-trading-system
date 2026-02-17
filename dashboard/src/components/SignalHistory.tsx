import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { db, TradeSignal, SignalStats } from '../lib/supabase';
import { useAuth } from '../context/AuthContext';
import './SignalHistory.css';

const COIN_CONFIG: Record<string, { icon: string; name: string; color: string }> = {
  'BTC_USDT': { icon: '\u20bf', name: 'Bitcoin', color: '#f7931a' },
  'ETH_USDT': { icon: '\u039e', name: 'Ethereum', color: '#627eea' },
  'SOL_USDT': { icon: '\u25ce', name: 'Solana', color: '#00ffa3' },
  'PEPE_USDT': { icon: '\ud83d\udc38', name: 'Pepe', color: '#4a9c2d' },
};

const SignalHistory: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [signals, setSignals] = useState<TradeSignal[]>([]);
  const [stats, setStats] = useState<SignalStats[]>([]);
  const [filterCoin, setFilterCoin] = useState<string>('');
  const [filterVerdict, setFilterVerdict] = useState<string>('');
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    if (!user) return;
    setLoading(true);
    try {
      const [signalData, statsData] = await Promise.all([
        db.getSignals(user.id, filterCoin || undefined, 100),
        db.getSignalStats(user.id)
      ]);
      setSignals(filterVerdict ? signalData.filter(s => s.verdict === filterVerdict) : signalData);
      setStats(statsData);
    } catch (err) {
      console.error('Error fetching signal history:', err);
    } finally {
      setLoading(false);
    }
  }, [user, filterCoin, filterVerdict]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const formatPrice = (p: number | null): string => {
    if (p === null) return '-';
    if (p < 0.01) return p.toFixed(6);
    if (p < 1) return p.toFixed(4);
    return '$' + p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };

  const totalSignals = stats.reduce((sum, s) => sum + s.total, 0);
  const totalBuys = stats.reduce((sum, s) => sum + s.buy, 0);
  const totalWaits = stats.reduce((sum, s) => sum + s.wait, 0);
  const totalAvoids = stats.reduce((sum, s) => sum + s.avoid, 0);

  return (
    <div className="signal-history-page">
      {/* Header */}
      <header className="signal-header">
        <div className="header-left">
          <button className="back-btn" onClick={() => navigate('/')}>Back</button>
          <h1>Signal History</h1>
          <span className="version">Prediction Tracking</span>
        </div>
      </header>

      {/* Stats Summary */}
      <div className="stats-section">
        <div className="stats-overview">
          <div className="stat-card total">
            <div className="stat-value">{totalSignals}</div>
            <div className="stat-label">Total Signals</div>
          </div>
          <div className="stat-card buy">
            <div className="stat-value">{totalBuys}</div>
            <div className="stat-label">BUY</div>
          </div>
          <div className="stat-card wait">
            <div className="stat-value">{totalWaits}</div>
            <div className="stat-label">WAIT</div>
          </div>
          <div className="stat-card avoid">
            <div className="stat-value">{totalAvoids}</div>
            <div className="stat-label">AVOID</div>
          </div>
        </div>

        {/* Per-coin breakdown */}
        {stats.length > 0 && (
          <div className="coin-stats-grid">
            {stats.map(s => {
              const config = COIN_CONFIG[s.coin];
              return (
                <div key={s.coin} className="coin-stat-card">
                  <div className="coin-stat-name" style={{ color: config?.color || '#fff' }}>
                    {config?.icon} {s.coin.replace('_USDT', '')}
                  </div>
                  <div className="coin-stat-total">{s.total} signals</div>
                  <div className="coin-stat-breakdown">
                    <span className="buy-count">{s.buy} BUY</span>
                    <span className="wait-count">{s.wait} WAIT</span>
                    <span className="avoid-count">{s.avoid} AVOID</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="signal-filters">
        <div className="filter-group">
          <label>Coin</label>
          <select value={filterCoin} onChange={e => setFilterCoin(e.target.value)} className="filter-select">
            <option value="">All Coins</option>
            {Object.keys(COIN_CONFIG).map(coin => (
              <option key={coin} value={coin}>{coin.replace('_USDT', '')}</option>
            ))}
          </select>
        </div>
        <div className="filter-group">
          <label>Verdict</label>
          <select value={filterVerdict} onChange={e => setFilterVerdict(e.target.value)} className="filter-select">
            <option value="">All</option>
            <option value="BUY">BUY</option>
            <option value="WAIT">WAIT</option>
            <option value="AVOID">AVOID</option>
            <option value="BLOCKED">BLOCKED</option>
          </select>
        </div>
      </div>

      {/* Signal Table */}
      {loading ? (
        <div className="signal-loading">
          <div className="spinner"></div>
          <p>Loading signal history...</p>
        </div>
      ) : signals.length === 0 ? (
        <div className="no-signals">
          <p>No signals recorded yet.</p>
          <p className="no-signals-hint">Run analysis on a coin page to start recording signals.</p>
        </div>
      ) : (
        <div className="signal-table-wrapper">
          <table className="signal-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Coin</th>
                <th>Type</th>
                <th>Verdict</th>
                <th>Win %</th>
                <th>Loss %</th>
                <th>Expectancy</th>
                <th>Price</th>
                <th>Regime</th>
              </tr>
            </thead>
            <tbody>
              {signals.map(s => (
                <tr key={s.id}>
                  <td className="signal-time">
                    {new Date(s.created_at).toLocaleDateString()}{' '}
                    {new Date(s.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </td>
                  <td>
                    <span style={{ color: COIN_CONFIG[s.coin]?.color }}>
                      {s.coin.replace('_USDT', '')}
                    </span>
                  </td>
                  <td>{s.trade_type || '-'}</td>
                  <td>
                    <span className={`verdict-badge verdict-${s.verdict.toLowerCase()}`}>
                      {s.verdict}
                    </span>
                  </td>
                  <td>{s.win_probability !== null ? `${s.win_probability.toFixed(1)}%` : '-'}</td>
                  <td>{s.loss_probability !== null ? `${s.loss_probability.toFixed(1)}%` : '-'}</td>
                  <td className={s.expectancy && s.expectancy > 0 ? 'positive' : s.expectancy && s.expectancy < 0 ? 'negative' : ''}>
                    {s.expectancy !== null ? `${s.expectancy.toFixed(1)}%` : '-'}
                  </td>
                  <td>{formatPrice(s.price_at_signal)}</td>
                  <td className="regime-cell">{s.market_regime || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default SignalHistory;
