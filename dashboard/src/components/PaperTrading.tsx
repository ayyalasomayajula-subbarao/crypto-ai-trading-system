import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createChart, ColorType, LineSeries } from 'lightweight-charts';
import type { IChartApi, LineData, Time } from 'lightweight-charts';
import axios from 'axios';
import './PaperTrading.css';

const API_BASE = process.env.REACT_APP_API_URL || window.location.origin;

interface PaperStatus {
  running: boolean;
  start_time: string;
  days_running: number;
  target_days: number;
  equity: number;
  initial_capital: number;
  total_return_pct: number;
  total_trades: number;
  open_positions: Record<string, Position>;
  coins: string[];
  config: Record<string, any>;
  last_processed: Record<string, string>;
}

interface Position {
  coin: string;
  entry_time: string;
  entry_price: number;
  tp_price: number;
  sl_price: number;
  position_size_usd: number;
  win_prob: number;
  loss_prob: number;
  candles_held: number;
}

interface Trade {
  coin: string;
  entry_time: string;
  exit_time: string;
  entry_price: number;
  exit_price: number;
  position_size_usd: number;
  win_prob: number;
  loss_prob: number;
  gross_pnl_pct: number;
  net_pnl_pct: number;
  pnl_usd: number;
  exit_reason: string;
  hours_held: number;
  result: string;
  equity_after: number;
}

interface Metrics {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  total_return_pct: number;
  avg_win_pct: number;
  avg_loss_pct: number;
  exit_reasons?: Record<string, number>;
  per_coin?: Record<string, { trades: number; win_rate: number; total_pnl_usd: number; avg_pnl_pct: number }>;
  equity_curve?: { timestamp: string; equity: number }[];
}

const formatPrice = (p: number): string => {
  if (p === 0) return '$0';
  if (p < 0.0001) return '$' + p.toFixed(8);
  if (p < 0.01) return '$' + p.toFixed(6);
  if (p < 1) return '$' + p.toFixed(4);
  if (p < 1000) return '$' + p.toFixed(2);
  return '$' + p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

const PaperTrading: React.FC = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  const [status, setStatus] = useState<PaperStatus | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [startCapital, setStartCapital] = useState(10000);

  const fetchData = useCallback(async () => {
    try {
      const [statusRes, tradesRes, metricsRes] = await Promise.all([
        axios.get(`${API_BASE}/paper-trading/status`),
        axios.get(`${API_BASE}/paper-trading/trades`),
        axios.get(`${API_BASE}/paper-trading/metrics`),
      ]);
      setStatus(statusRes.data);
      setTrades(tradesRes.data.trades || []);
      setMetrics(metricsRes.data);
    } catch (err) {
      console.error('Error fetching paper trading data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // refresh every 30s
    return () => clearInterval(interval);
  }, [fetchData]);

  // Equity chart
  useEffect(() => {
    if (!chartContainerRef.current || !metrics?.equity_curve?.length) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#64748b',
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.03)' },
        horzLines: { color: 'rgba(255,255,255,0.03)' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 300,
      timeScale: { timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: 'rgba(255,255,255,0.1)' },
    });

    const lineSeries = chart.addSeries(LineSeries, {
      color: '#3b82f6',
      lineWidth: 2,
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    });

    const data: LineData[] = metrics.equity_curve.map((point) => ({
      time: (new Date(point.timestamp).getTime() / 1000) as Time,
      value: point.equity,
    }));

    lineSeries.setData(data);
    chart.timeScale().fitContent();
    chartRef.current = chart;

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [metrics?.equity_curve]);

  const handleStart = async () => {
    setActionLoading(true);
    try {
      await axios.post(`${API_BASE}/paper-trading/start?capital=${startCapital}`);
      await fetchData();
    } catch (err) {
      console.error('Error starting paper trading:', err);
    } finally {
      setActionLoading(false);
    }
  };

  const handleStop = async () => {
    setActionLoading(true);
    try {
      await axios.post(`${API_BASE}/paper-trading/stop`);
      await fetchData();
    } catch (err) {
      console.error('Error stopping paper trading:', err);
    } finally {
      setActionLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="paper-trading-page">
        <div className="loading-state">Loading paper trading data...</div>
      </div>
    );
  }

  const isRunning = status?.running || false;
  const daysRunning = status?.days_running || 0;
  const targetDays = status?.target_days || 45;
  const progressPct = Math.min((daysRunning / targetDays) * 100, 100);
  const equity = status?.equity || 10000;
  const initialCapital = status?.initial_capital || 10000;
  const totalReturn = status?.total_return_pct || 0;
  const openPositions = status?.open_positions || {};

  return (
    <div className="paper-trading-page">
      {/* Header */}
      <div className="paper-header">
        <div className="header-left">
          <h1>Paper Trading</h1>
          <span className={`status-badge ${isRunning ? 'running' : 'stopped'}`}>
            {isRunning ? 'RUNNING' : 'STOPPED'}
          </span>
        </div>
        <div className="header-actions">
          {!isRunning && !status?.total_trades && (
            <div className="capital-input-group">
              <label>Capital $</label>
              <input
                type="number"
                value={startCapital}
                onChange={(e) => setStartCapital(Number(e.target.value))}
                min={100}
                step={1000}
                className="capital-input"
              />
            </div>
          )}
          {!isRunning ? (
            <button className="control-btn start" onClick={handleStart} disabled={actionLoading}>
              {actionLoading ? 'Starting...' : 'Start Trading'}
            </button>
          ) : (
            <button className="control-btn stop" onClick={handleStop} disabled={actionLoading}>
              {actionLoading ? 'Stopping...' : 'Stop Trading'}
            </button>
          )}
        </div>
      </div>

      <div className="paper-content">
        {/* Day Progress */}
        <div className="day-progress">
          <div className="progress-header">
            <h3>Experiment Progress</h3>
            <span>Day {daysRunning} of {targetDays}</span>
          </div>
          <div className="progress-bar-container">
            <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
          </div>
        </div>

        {/* Top Stats */}
        <div className="stats-row">
          <div className="stat-card">
            <div className="stat-label">Equity</div>
            <div className={`stat-value ${equity >= initialCapital ? 'positive' : 'negative'}`}>
              ${equity.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </div>
            <div className="stat-sub">Initial: ${initialCapital.toLocaleString()}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Total Return</div>
            <div className={`stat-value ${totalReturn >= 0 ? 'positive' : 'negative'}`}>
              {totalReturn >= 0 ? '+' : ''}{totalReturn.toFixed(2)}%
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Total Trades</div>
            <div className="stat-value">{metrics?.total_trades || 0}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Win Rate</div>
            <div className={`stat-value ${(metrics?.win_rate || 0) >= 55 ? 'positive' : (metrics?.win_rate || 0) >= 45 ? 'neutral' : 'negative'}`}>
              {(metrics?.win_rate || 0).toFixed(1)}%
            </div>
            <div className="stat-sub">Target: 55%+</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Profit Factor</div>
            <div className={`stat-value ${(metrics?.profit_factor || 0) >= 1.5 ? 'positive' : 'neutral'}`}>
              {(metrics?.profit_factor || 0).toFixed(2)}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Max Drawdown</div>
            <div className={`stat-value ${(metrics?.max_drawdown_pct || 0) >= -15 ? 'positive' : 'negative'}`}>
              {(metrics?.max_drawdown_pct || 0).toFixed(2)}%
            </div>
            <div className="stat-sub">Limit: -15%</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Sharpe Ratio</div>
            <div className={`stat-value ${(metrics?.sharpe_ratio || 0) >= 1 ? 'positive' : 'neutral'}`}>
              {(metrics?.sharpe_ratio || 0).toFixed(2)}
            </div>
          </div>
        </div>

        {/* Equity Chart */}
        {metrics?.equity_curve && metrics.equity_curve.length > 1 && (
          <div className="equity-chart-section">
            <h3>Equity Curve</h3>
            <div className="equity-chart-container" ref={chartContainerRef} />
          </div>
        )}

        {/* Open Positions + Per-Coin Metrics */}
        <div className="panel-grid">
          <div className="panel">
            <h3>Open Positions ({Object.keys(openPositions).length})</h3>
            {Object.keys(openPositions).length === 0 ? (
              <div className="empty-state">No open positions</div>
            ) : (
              <table className="positions-table">
                <thead>
                  <tr>
                    <th>Coin</th>
                    <th>Entry</th>
                    <th>TP</th>
                    <th>SL</th>
                    <th>Size</th>
                    <th>Hours</th>
                    <th>Win%</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(openPositions).map(([coin, pos]) => (
                    <tr key={coin}>
                      <td style={{ fontWeight: 700 }}>{coin.replace('_USDT', '')}</td>
                      <td>{formatPrice(pos.entry_price)}</td>
                      <td style={{ color: '#22c55e' }}>{formatPrice(pos.tp_price)}</td>
                      <td style={{ color: '#ef4444' }}>{formatPrice(pos.sl_price)}</td>
                      <td>${pos.position_size_usd.toFixed(0)}</td>
                      <td>{pos.candles_held}h</td>
                      <td>{pos.win_prob}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="panel">
            <h3>Per-Coin Breakdown</h3>
            {!metrics?.per_coin || Object.keys(metrics.per_coin).length === 0 ? (
              <div className="empty-state">No trades yet</div>
            ) : (
              <div className="coin-metrics-grid">
                {Object.entries(metrics.per_coin).map(([coin, data]) => (
                  <div className="coin-metric-card" key={coin}>
                    <div className="coin-name">{coin.replace('_USDT', '')}</div>
                    <div className="metric-row">
                      <span className="label">Trades</span>
                      <span className="value">{data.trades}</span>
                    </div>
                    <div className="metric-row">
                      <span className="label">Win Rate</span>
                      <span className="value" style={{ color: data.win_rate >= 55 ? '#22c55e' : '#f59e0b' }}>
                        {data.win_rate.toFixed(1)}%
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="label">Total PnL</span>
                      <span className="value" style={{ color: data.total_pnl_usd >= 0 ? '#22c55e' : '#ef4444' }}>
                        ${data.total_pnl_usd.toFixed(2)}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="label">Avg PnL</span>
                      <span className="value">{data.avg_pnl_pct.toFixed(2)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Frozen Config */}
        <div className="panel">
          <h3>Frozen Configuration (Locked)</h3>
          <div className="config-grid">
            <div className="config-item">
              <span className="config-key">Coins</span>
              <span className="config-val">{status?.coins?.join(', ') || 'SOL, PEPE'}</span>
            </div>
            <div className="config-item">
              <span className="config-key">Take Profit</span>
              <span className="config-val">{((status?.config?.tp_pct || 0.05) * 100).toFixed(0)}%</span>
            </div>
            <div className="config-item">
              <span className="config-key">Stop Loss</span>
              <span className="config-val">{((status?.config?.sl_pct || 0.03) * 100).toFixed(0)}%</span>
            </div>
            <div className="config-item">
              <span className="config-key">Time Limit</span>
              <span className="config-val">{status?.config?.time_limit_hours || 48}h</span>
            </div>
            <div className="config-item">
              <span className="config-key">Risk/Trade</span>
              <span className="config-val">{((status?.config?.risk_per_trade || 0.005) * 100).toFixed(1)}%</span>
            </div>
            <div className="config-item">
              <span className="config-key">Round-Trip Cost</span>
              <span className="config-val">{((status?.config?.round_trip_cost || 0.0022) * 100).toFixed(2)}%</span>
            </div>
            {status?.config?.thresholds && Object.entries(status.config.thresholds).map(([coin, thresh]) => (
              <div className="config-item" key={coin}>
                <span className="config-key">{coin.replace('_USDT', '')} Threshold</span>
                <span className="config-val">{(Number(thresh) * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Trade Log */}
        <div className="trade-log-section">
          <h3>Trade Log ({trades.length} trades)</h3>
          {trades.length === 0 ? (
            <div className="empty-state">No trades recorded yet. Start paper trading and wait for signals.</div>
          ) : (
            <table className="trades-table">
              <thead>
                <tr>
                  <th>Coin</th>
                  <th>Entry Time</th>
                  <th>Exit Time</th>
                  <th>Entry</th>
                  <th>Exit</th>
                  <th>Size</th>
                  <th>PnL %</th>
                  <th>PnL $</th>
                  <th>Exit</th>
                  <th>Hours</th>
                  <th>Result</th>
                  <th>Equity</th>
                </tr>
              </thead>
              <tbody>
                {[...trades].reverse().map((t, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 700 }}>{t.coin.replace('_USDT', '')}</td>
                    <td style={{ fontSize: '0.72rem' }}>{new Date(t.entry_time).toLocaleString()}</td>
                    <td style={{ fontSize: '0.72rem' }}>{new Date(t.exit_time).toLocaleString()}</td>
                    <td>{formatPrice(t.entry_price)}</td>
                    <td>{formatPrice(t.exit_price)}</td>
                    <td>${t.position_size_usd.toFixed(0)}</td>
                    <td className={t.net_pnl_pct >= 0 ? 'win' : 'loss'}>
                      {t.net_pnl_pct >= 0 ? '+' : ''}{t.net_pnl_pct.toFixed(2)}%
                    </td>
                    <td className={t.pnl_usd >= 0 ? 'win' : 'loss'}>
                      {t.pnl_usd >= 0 ? '+' : ''}${t.pnl_usd.toFixed(2)}
                    </td>
                    <td><span className={`exit-badge ${t.exit_reason}`}>{t.exit_reason}</span></td>
                    <td>{t.hours_held}h</td>
                    <td className={t.result === 'WIN' ? 'win' : 'loss'} style={{ fontWeight: 700 }}>
                      {t.result}
                    </td>
                    <td>${t.equity_after.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Exit Reason Distribution */}
        {metrics?.exit_reasons && Object.keys(metrics.exit_reasons).length > 0 && (
          <div className="panel">
            <h3>Exit Reason Distribution</h3>
            <div className="config-grid">
              {Object.entries(metrics.exit_reasons).map(([reason, count]) => (
                <div className="config-item" key={reason}>
                  <span className="config-key">
                    <span className={`exit-badge ${reason}`}>{reason}</span>
                  </span>
                  <span className="config-val">
                    {count} ({((count as number) / (metrics.total_trades || 1) * 100).toFixed(0)}%)
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PaperTrading;
