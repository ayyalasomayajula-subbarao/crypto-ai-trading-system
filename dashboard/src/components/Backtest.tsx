import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { createChart, ColorType, LineSeries } from 'lightweight-charts';
import type { IChartApi, LineData, Time } from 'lightweight-charts';
import axios from 'axios';
import './Backtest.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const formatPrice = (p: number): string => {
  if (p === 0) return '$0';
  if (p < 0.0001) return '$' + p.toFixed(8);
  if (p < 0.01) return '$' + p.toFixed(6);
  if (p < 1) return '$' + p.toFixed(4);
  if (p < 1000) return '$' + p.toFixed(2);
  return '$' + p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

const COINS = [
  { id: 'ALL', label: 'All Coins' },
  { id: 'BTC_USDT', label: 'BTC', color: '#f7931a' },
  { id: 'ETH_USDT', label: 'ETH', color: '#627eea' },
  { id: 'SOL_USDT', label: 'SOL', color: '#00ffa3' },
  { id: 'PEPE_USDT', label: 'PEPE', color: '#4a9c2d' },
];

interface BacktestMetrics {
  initial_capital: number;
  final_capital: number;
  total_return_pct: number;
  annualized_return_pct: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  avg_win_pct: number;
  avg_loss_pct: number;
  profit_factor: number;
  max_drawdown_pct: number;
  sharpe_ratio: number;
  tp_exits: number;
  sl_exits: number;
  time_exits: number;
}

interface EquityPoint {
  timestamp: string;
  equity: number;
}

interface Trade {
  entry_time: string;
  exit_time: string;
  entry_price: number;
  exit_price: number;
  win_prob: number;
  net_pnl_pct: number;
  pnl_usd: number;
  exit_reason: string;
  hours_held: number;
  result: string;
  regime: string;
}

interface RegimeStats {
  [regime: string]: { trades: number; wins: number; win_rate: number };
}

interface BacktestResult {
  coin: string;
  metrics: BacktestMetrics | null;
  equity_curve: EquityPoint[];
  trades: Trade[];
  regime_stats: RegimeStats;
  config: Record<string, number>;
  test_period?: { start: string; end: string; days: number; rows: number };
  error?: string;
}

interface WFValidationResult {
  coin: string;
  splits: {
    train: [string, string];
    validate: [string, string];
    test: [string, string];
  };
  train_stats: {
    rows: number;
    accuracy: number;
    label_distribution: Record<string, number>;
  };
  validation: {
    rows: number;
    threshold_results: Array<{
      threshold: number;
      trades: number;
      win_rate: number;
      sharpe: number;
      return_pct: number;
      profit_factor: number;
      max_dd: number;
    }>;
    chosen_threshold: number;
    chosen_sharpe: number;
  };
  test: {
    rows: number;
    threshold: number;
    metrics: BacktestMetrics & { initial_capital: number; final_capital: number };
    trades: Trade[];
  };
  verdict: string;
  error?: string;
}

interface RobustnessWindow {
  label: string;
  train_period?: string;
  test_period?: string;
  train_rows?: number;
  test_rows?: number;
  threshold?: number;
  metrics?: {
    total_trades: number;
    win_rate: number;
    profit_factor: number;
    total_return_pct: number;
    max_drawdown_pct: number;
    sharpe_ratio: number;
  };
  error?: string;
}

interface RobustnessResult {
  coin: string;
  windows: RobustnessWindow[];
  verdict: string;
  robustness_details: {
    profitable_years: number;
    total_years?: number;
    avg_pf: number;
    worst_dd: number;
    pass_3_years?: boolean;
    pass_pf?: boolean;
    pass_dd?: boolean;
    reason?: string;
  };
  error?: string;
}

interface MonteCarloResult {
  coin: string;
  simulations: number;
  base_trades: number;
  summary: {
    profitable_pct: number;
    median_return_pct: number;
    mean_return_pct: number;
    worst_return_pct: number;
    best_return_pct: number;
    std_return_pct: number;
    percentile_5: number;
    percentile_25: number;
    percentile_75: number;
    percentile_95: number;
  };
  pass_criteria: {
    profitable_70pct: boolean;
    median_positive: boolean;
    worst_case_above_neg10: boolean;
  };
  verdict: string;
  distribution: number[];
}

const Backtest: React.FC = () => {
  const navigate = useNavigate();
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  const [selectedCoin, setSelectedCoin] = useState('BTC_USDT');
  const [threshold, setThreshold] = useState(0.45);
  const [capital, setCapital] = useState(10000);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [allResults, setAllResults] = useState<Record<string, BacktestResult> | null>(null);
  const [showTrades, setShowTrades] = useState(false);
  const [error, setError] = useState('');
  const [mcResult, setMcResult] = useState<MonteCarloResult | null>(null);
  const [mcLoading, setMcLoading] = useState(false);
  const [wfResult, setWfResult] = useState<WFValidationResult | null>(null);
  const [wfLoading, setWfLoading] = useState(false);
  const [robResult, setRobResult] = useState<RobustnessResult | null>(null);
  const [robLoading, setRobLoading] = useState(false);

  const runBacktest = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    setAllResults(null);

    try {
      if (selectedCoin === 'ALL') {
        const resp = await axios.get(`${API_BASE}/backtest`, {
          params: { threshold, capital }
        });
        setAllResults(resp.data.results);
        // Show first coin with results
        const firstCoin = Object.keys(resp.data.results).find(
          k => resp.data.results[k]?.metrics
        );
        if (firstCoin) setResult(resp.data.results[firstCoin]);
      } else {
        const resp = await axios.get(`${API_BASE}/backtest/${selectedCoin}`, {
          params: { threshold, capital }
        });
        setResult(resp.data);
      }
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Backtest failed');
    } finally {
      setLoading(false);
    }
  };

  const runMonteCarlo = async () => {
    const coin = selectedCoin === 'ALL' ? 'BTC_USDT' : selectedCoin;
    setMcLoading(true);
    setMcResult(null);
    try {
      const resp = await axios.get(`${API_BASE}/backtest/monte-carlo/${coin}`, {
        params: { simulations: 100, threshold }
      });
      setMcResult(resp.data);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Monte Carlo failed');
    } finally {
      setMcLoading(false);
    }
  };

  const runWalkForward = async () => {
    const coin = selectedCoin === 'ALL' ? 'BTC_USDT' : selectedCoin;
    setWfLoading(true);
    setWfResult(null);
    try {
      const resp = await axios.get(`${API_BASE}/backtest/walk-forward-validation/${coin}`);
      setWfResult(resp.data);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Walk-forward validation failed');
    } finally {
      setWfLoading(false);
    }
  };

  const runRobustness = async () => {
    const coin = selectedCoin === 'ALL' ? 'BTC_USDT' : selectedCoin;
    setRobLoading(true);
    setRobResult(null);
    try {
      const resp = await axios.get(`${API_BASE}/backtest/rolling-robustness/${coin}`);
      setRobResult(resp.data);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Robustness test failed');
    } finally {
      setRobLoading(false);
    }
  };

  // Render equity curve chart
  const renderChart = useCallback(() => {
    if (!chartContainerRef.current || !result?.equity_curve?.length) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const container = chartContainerRef.current;
    const coinColor = COINS.find(c => c.id === result.coin)?.color || '#00e676';

    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#888',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#1a1a2e' },
        horzLines: { color: '#1a1a2e' },
      },
      rightPriceScale: {
        borderColor: '#222',
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      timeScale: {
        borderColor: '#222',
        timeVisible: true,
      },
      width: container.clientWidth,
      height: 350,
    });

    chartRef.current = chart;

    const series = chart.addSeries(LineSeries, {
      color: coinColor,
      lineWidth: 2,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
    });

    const chartData: LineData<Time>[] = result.equity_curve.map(p => ({
      time: (new Date(p.timestamp).getTime() / 1000) as Time,
      value: p.equity,
    }));

    series.setData(chartData);
    chart.timeScale().fitContent();

    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        if (chartRef.current) {
          chartRef.current.applyOptions({ width: entry.contentRect.width });
        }
      }
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [result]);

  useEffect(() => {
    const cleanup = renderChart();
    return () => cleanup?.();
  }, [renderChart]);

  const m = result?.metrics;

  return (
    <div className="backtest-page">
      {/* Header */}
      <header className="backtest-header">
        <div className="header-left">
          <button className="back-btn" onClick={() => navigate('/')}>Back</button>
          <h1>Backtesting Engine</h1>
          <span className="version">Walk-Forward Validation</span>
        </div>
      </header>

      {/* Controls */}
      <div className="backtest-controls">
        <div className="control-group">
          <label>Coin</label>
          <div className="coin-selector">
            {COINS.map(c => (
              <button
                key={c.id}
                className={`coin-btn ${selectedCoin === c.id ? 'active' : ''}`}
                style={selectedCoin === c.id && c.color ? { borderColor: c.color, color: c.color } : {}}
                onClick={() => setSelectedCoin(c.id)}
              >
                {c.label}
              </button>
            ))}
          </div>
        </div>

        <div className="control-group">
          <label>WIN Threshold: {(threshold * 100).toFixed(0)}%</label>
          <input
            type="range"
            min="0.35"
            max="0.60"
            step="0.01"
            value={threshold}
            onChange={e => setThreshold(parseFloat(e.target.value))}
            className="slider"
          />
        </div>

        <div className="control-group">
          <label>Capital ($)</label>
          <input
            type="number"
            value={capital}
            onChange={e => setCapital(parseInt(e.target.value) || 10000)}
            className="capital-input"
          />
        </div>

        <button className="run-btn" onClick={runBacktest} disabled={loading}>
          {loading ? 'Running...' : 'Run Backtest'}
        </button>
      </div>

      {error && <div className="backtest-error">{error}</div>}

      {loading && (
        <div className="backtest-loading">
          <div className="spinner"></div>
          <p>Running backtest on historical data...</p>
        </div>
      )}

      {/* All Coins Summary (when ALL selected) */}
      {allResults && (
        <div className="all-coins-summary">
          <h3>All Coins Summary</h3>
          <div className="summary-grid">
            {Object.entries(allResults).map(([coin, r]) => (
              <div
                key={coin}
                className={`summary-card ${result?.coin === coin ? 'active' : ''}`}
                onClick={() => setResult(r)}
              >
                <div className="summary-coin" style={{ color: COINS.find(c => c.id === coin)?.color }}>
                  {coin.replace('_USDT', '')}
                </div>
                {r.metrics ? (
                  <>
                    <div className={`summary-return ${r.metrics.total_return_pct >= 0 ? 'positive' : 'negative'}`}>
                      {r.metrics.total_return_pct >= 0 ? '+' : ''}{r.metrics.total_return_pct.toFixed(1)}%
                    </div>
                    <div className="summary-detail">
                      WR: {r.metrics.win_rate}% | Sharpe: {r.metrics.sharpe_ratio}
                    </div>
                  </>
                ) : (
                  <div className="summary-detail">No trades</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Metrics Cards */}
      {m && (
        <div className="metrics-section">
          <div className="metrics-grid">
            <div className="metric-card highlight">
              <div className="metric-label">Total Return</div>
              <div className={`metric-value ${m.total_return_pct >= 0 ? 'positive' : 'negative'}`}>
                {m.total_return_pct >= 0 ? '+' : ''}{m.total_return_pct.toFixed(2)}%
              </div>
              <div className="metric-sub">${m.initial_capital.toLocaleString()} → ${m.final_capital.toLocaleString()}</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Win Rate</div>
              <div className={`metric-value ${m.win_rate >= 55 ? 'positive' : m.win_rate >= 45 ? 'neutral' : 'negative'}`}>
                {m.win_rate}%
              </div>
              <div className="metric-sub">{m.winning_trades}W / {m.losing_trades}L</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Sharpe Ratio</div>
              <div className={`metric-value ${m.sharpe_ratio >= 1 ? 'positive' : m.sharpe_ratio >= 0 ? 'neutral' : 'negative'}`}>
                {m.sharpe_ratio.toFixed(2)}
              </div>
              <div className="metric-sub">{m.sharpe_ratio >= 2 ? 'Excellent' : m.sharpe_ratio >= 1 ? 'Good' : m.sharpe_ratio >= 0 ? 'Fair' : 'Poor'}</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Max Drawdown</div>
              <div className="metric-value negative">{m.max_drawdown_pct.toFixed(2)}%</div>
              <div className="metric-sub">Largest peak-to-trough</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Profit Factor</div>
              <div className={`metric-value ${m.profit_factor >= 1.5 ? 'positive' : m.profit_factor >= 1 ? 'neutral' : 'negative'}`}>
                {m.profit_factor.toFixed(2)}
              </div>
              <div className="metric-sub">Gross profit / loss</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Total Trades</div>
              <div className="metric-value">{m.total_trades}</div>
              <div className="metric-sub">TP: {m.tp_exits} | SL: {m.sl_exits} | Time: {m.time_exits}</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Avg Win</div>
              <div className="metric-value positive">+{m.avg_win_pct.toFixed(2)}%</div>
              <div className="metric-sub">Per winning trade</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Avg Loss</div>
              <div className="metric-value negative">{m.avg_loss_pct.toFixed(2)}%</div>
              <div className="metric-sub">Per losing trade</div>
            </div>
          </div>
        </div>
      )}

      {/* Equity Curve */}
      {result && result.equity_curve && result.equity_curve.length > 0 && (
        <div className="equity-section">
          <h3>Equity Curve - {result.coin.replace('_', '/')}</h3>
          {result.test_period && (
            <span className="test-period">
              {new Date(result.test_period.start).toLocaleDateString()} - {new Date(result.test_period.end).toLocaleDateString()}
              ({result.test_period.days} days)
            </span>
          )}
          <div className="equity-chart" ref={chartContainerRef}></div>
        </div>
      )}

      {/* Regime Stats */}
      {result?.regime_stats && Object.keys(result.regime_stats).length > 0 && (
        <div className="regime-section">
          <h3>Win Rate by Market Regime</h3>
          <div className="regime-grid">
            {Object.entries(result.regime_stats).map(([regime, stats]) => (
              <div key={regime} className="regime-card">
                <div className="regime-name">{regime.replace('_', ' ')}</div>
                <div className={`regime-winrate ${stats.win_rate >= 55 ? 'positive' : stats.win_rate >= 45 ? 'neutral' : 'negative'}`}>
                  {stats.win_rate}%
                </div>
                <div className="regime-trades">{stats.trades} trades</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Trade Log */}
      {result && result.trades && result.trades.length > 0 && (
        <div className="trades-section">
          <button className="toggle-trades" onClick={() => setShowTrades(!showTrades)}>
            {showTrades ? 'Hide' : 'Show'} Trade Log ({result.trades.length} trades)
          </button>

          {showTrades && (
            <div className="trades-table-wrapper">
              <table className="trades-table">
                <thead>
                  <tr>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Entry $</th>
                    <th>Exit $</th>
                    <th>Win Prob</th>
                    <th>PnL %</th>
                    <th>PnL $</th>
                    <th>Exit</th>
                    <th>Hours</th>
                    <th>Regime</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trades.map((t, i) => (
                    <tr key={i} className={t.result === 'WIN' ? 'trade-win' : 'trade-loss'}>
                      <td>{new Date(t.entry_time).toLocaleDateString()}</td>
                      <td>{new Date(t.exit_time).toLocaleDateString()}</td>
                      <td>{formatPrice(t.entry_price)}</td>
                      <td>{formatPrice(t.exit_price)}</td>
                      <td>{t.win_prob}%</td>
                      <td className={t.net_pnl_pct >= 0 ? 'positive' : 'negative'}>
                        {t.net_pnl_pct >= 0 ? '+' : ''}{t.net_pnl_pct}%
                      </td>
                      <td className={t.pnl_usd >= 0 ? 'positive' : 'negative'}>
                        ${t.pnl_usd.toFixed(2)}
                      </td>
                      <td><span className={`exit-badge ${t.exit_reason.toLowerCase()}`}>{t.exit_reason}</span></td>
                      <td>{t.hours_held}h</td>
                      <td>{t.regime}</td>
                      <td><span className={`result-badge ${t.result.toLowerCase()}`}>{t.result}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Config / Cost Breakdown */}
      {result?.config && (
        <div className="config-section">
          <h3>Trading Cost Model</h3>
          <div className="config-grid">
            <div className="config-item">
              <span className="config-label">Exchange Fee</span>
              <span className="config-value">{result.config.fee_pct?.toFixed(2) || '0.06'}% / side</span>
            </div>
            <div className="config-item">
              <span className="config-label">Slippage</span>
              <span className="config-value">{result.config.slippage_pct?.toFixed(2) || '0.03'}% / side</span>
            </div>
            <div className="config-item">
              <span className="config-label">Spread</span>
              <span className="config-value">{result.config.spread_pct?.toFixed(2) || '0.02'}% / side</span>
            </div>
            <div className="config-item highlight">
              <span className="config-label">Total Cost / Trade</span>
              <span className="config-value">{result.config.total_cost_per_trade?.toFixed(3) || '0.22'}%</span>
            </div>
            <div className="config-item">
              <span className="config-label">TP / SL</span>
              <span className="config-value">{result.config.tp_pct}% / {result.config.sl_pct}%</span>
            </div>
            <div className="config-item">
              <span className="config-label">Position Size</span>
              <span className="config-value">{result.config.position_size_pct}%</span>
            </div>
          </div>
        </div>
      )}

      {/* Walk-Forward Validation Section */}
      <div className="wf-validation-section">
        <div className="wf-header">
          <h3>TRUE Walk-Forward Validation</h3>
          <span className="wf-subtitle">No data leakage. Model retrained. Single final test.</span>
          <button
            className="run-btn wf-btn"
            onClick={runWalkForward}
            disabled={wfLoading || selectedCoin === 'ALL'}
          >
            {wfLoading ? 'Training & Testing...' : 'Run Walk-Forward'}
          </button>
        </div>

        {wfLoading && (
          <div className="backtest-loading">
            <div className="spinner"></div>
            <p>Retraining model on train block, tuning threshold, running final test...</p>
          </div>
        )}

        {wfResult && !wfResult.error && wfResult.test && (
          <div className="wf-results">
            {/* Verdict Banner */}
            <div className={`wf-verdict ${wfResult.verdict.toLowerCase().replace('_', '-')}`}>
              {wfResult.verdict === 'VIABLE' ? 'VIABLE' :
               wfResult.verdict === 'MARGINAL' ? 'MARGINAL' :
               wfResult.verdict === 'INSUFFICIENT_DATA' ? 'INSUFFICIENT DATA' : 'NOT VIABLE'}
              <span className="wf-verdict-sub">
                {wfResult.verdict === 'VIABLE' ? 'Strategy shows edge on unseen data' :
                 wfResult.verdict === 'MARGINAL' ? 'Weak edge — needs more validation' :
                 wfResult.verdict === 'INSUFFICIENT_DATA' ? 'Too few trades for statistical significance' :
                 'No reliable edge detected'}
              </span>
            </div>

            {/* 3-Block Split Visual */}
            <div className="wf-splits">
              <div className="wf-split train">
                <div className="split-label">TRAIN</div>
                <div className="split-dates">{wfResult.splits.train[0]} → {wfResult.splits.train[1]}</div>
                <div className="split-detail">{wfResult.train_stats.rows.toLocaleString()} rows | Acc: {wfResult.train_stats.accuracy}%</div>
              </div>
              <div className="wf-split validate">
                <div className="split-label">VALIDATE</div>
                <div className="split-dates">{wfResult.splits.validate[0]} → {wfResult.splits.validate[1]}</div>
                <div className="split-detail">Threshold: {wfResult.validation.chosen_threshold} (Sharpe: {wfResult.validation.chosen_sharpe})</div>
              </div>
              <div className="wf-split test">
                <div className="split-label">FINAL TEST</div>
                <div className="split-dates">{wfResult.splits.test[0]} → {wfResult.splits.test[1]}</div>
                <div className="split-detail">{wfResult.test.rows.toLocaleString()} rows | Never touched until now</div>
              </div>
            </div>

            {/* Validation Threshold Selection */}
            <div className="wf-threshold-table">
              <h4>Threshold Selection (Validation Set)</h4>
              <table className="trades-table">
                <thead>
                  <tr>
                    <th>Threshold</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Sharpe</th>
                    <th>Return</th>
                    <th>PF</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {wfResult.validation.threshold_results.map((r) => (
                    <tr key={r.threshold} className={r.threshold === wfResult.validation.chosen_threshold ? 'chosen-row' : ''}>
                      <td>{(r.threshold * 100).toFixed(0)}%</td>
                      <td>{r.trades}</td>
                      <td>{r.win_rate.toFixed(1)}%</td>
                      <td>{r.sharpe.toFixed(2)}</td>
                      <td className={r.return_pct >= 0 ? 'positive' : 'negative'}>
                        {r.return_pct >= 0 ? '+' : ''}{r.return_pct.toFixed(2)}%
                      </td>
                      <td>{r.profit_factor.toFixed(2)}</td>
                      <td>{r.threshold === wfResult.validation.chosen_threshold ? 'CHOSEN' : ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Final Test Metrics */}
            <div className="wf-final-metrics">
              <h4>Final Test Results (Unseen Data)</h4>
              <div className="metrics-grid">
                <div className="metric-card highlight">
                  <div className="metric-label">Total Return</div>
                  <div className={`metric-value ${wfResult.test.metrics.total_return_pct >= 0 ? 'positive' : 'negative'}`}>
                    {wfResult.test.metrics.total_return_pct >= 0 ? '+' : ''}{wfResult.test.metrics.total_return_pct.toFixed(2)}%
                  </div>
                  <div className="metric-sub">${wfResult.test.metrics.initial_capital.toLocaleString()} → ${wfResult.test.metrics.final_capital.toLocaleString()}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Win Rate</div>
                  <div className={`metric-value ${wfResult.test.metrics.win_rate >= 55 ? 'positive' : wfResult.test.metrics.win_rate >= 45 ? 'neutral' : 'negative'}`}>
                    {wfResult.test.metrics.win_rate}%
                  </div>
                  <div className="metric-sub">{wfResult.test.metrics.winning_trades}W / {wfResult.test.metrics.losing_trades}L</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Sharpe Ratio</div>
                  <div className={`metric-value ${wfResult.test.metrics.sharpe_ratio >= 1 ? 'positive' : wfResult.test.metrics.sharpe_ratio >= 0 ? 'neutral' : 'negative'}`}>
                    {wfResult.test.metrics.sharpe_ratio.toFixed(2)}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Profit Factor</div>
                  <div className={`metric-value ${wfResult.test.metrics.profit_factor >= 1.5 ? 'positive' : wfResult.test.metrics.profit_factor >= 1 ? 'neutral' : 'negative'}`}>
                    {wfResult.test.metrics.profit_factor.toFixed(2)}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Max Drawdown</div>
                  <div className="metric-value negative">{wfResult.test.metrics.max_drawdown_pct.toFixed(2)}%</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Total Trades</div>
                  <div className="metric-value">{wfResult.test.metrics.total_trades}</div>
                  <div className="metric-sub">TP: {wfResult.test.metrics.tp_exits} | SL: {wfResult.test.metrics.sl_exits} | Time: {wfResult.test.metrics.time_exits}</div>
                </div>
              </div>
            </div>

            {/* Label Distribution */}
            <div className="wf-label-dist">
              <h4>Training Label Distribution</h4>
              <div className="label-bars">
                {Object.entries(wfResult.train_stats.label_distribution).map(([label, pct]) => (
                  <div key={label} className="label-bar-item">
                    <span className={`label-name ${label.toLowerCase()}`}>{label}</span>
                    <div className="label-bar-track">
                      <div className={`label-bar-fill ${label.toLowerCase()}`} style={{ width: `${pct}%` }}></div>
                    </div>
                    <span className="label-pct">{pct.toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {wfResult && wfResult.error && (
          <div className="backtest-error">{wfResult.error}</div>
        )}
      </div>

      {/* Rolling Robustness Section */}
      <div className="robustness-section">
        <div className="rob-header">
          <h3>Rolling Walk-Forward Robustness</h3>
          <span className="rob-subtitle">Multiple market regimes. Expanding windows.</span>
          <button
            className="run-btn rob-btn"
            onClick={runRobustness}
            disabled={robLoading || selectedCoin === 'ALL'}
          >
            {robLoading ? 'Testing all windows...' : 'Run Robustness'}
          </button>
        </div>

        {robLoading && (
          <div className="backtest-loading">
            <div className="spinner"></div>
            <p>Training models across multiple years...</p>
          </div>
        )}

        {robResult && !robResult.error && robResult.windows && (
          <div className="rob-results">
            {/* Verdict Banner */}
            <div className={`rob-verdict ${robResult.verdict.toLowerCase().replace('_', '-')}`}>
              {robResult.verdict}
              <span className="rob-verdict-sub">
                {robResult.verdict === 'PASS'
                  ? `${robResult.robustness_details.profitable_years}/${robResult.robustness_details.total_years} years profitable | Avg PF: ${robResult.robustness_details.avg_pf} | Worst DD: ${robResult.robustness_details.worst_dd}%`
                  : robResult.robustness_details.reason || 'Insufficient data for robustness determination'}
              </span>
            </div>

            {/* Pass/Fail Criteria */}
            {robResult.robustness_details.pass_3_years !== undefined && (
              <div className="rob-criteria">
                <div className={`criteria-item ${robResult.robustness_details.pass_3_years ? 'pass' : 'fail'}`}>
                  {robResult.robustness_details.pass_3_years ? 'PASS' : 'FAIL'} 3+ years profitable ({robResult.robustness_details.profitable_years}/{robResult.robustness_details.total_years})
                </div>
                <div className={`criteria-item ${robResult.robustness_details.pass_pf ? 'pass' : 'fail'}`}>
                  {robResult.robustness_details.pass_pf ? 'PASS' : 'FAIL'} Avg PF &ge; 1.8 ({robResult.robustness_details.avg_pf})
                </div>
                <div className={`criteria-item ${robResult.robustness_details.pass_dd ? 'pass' : 'fail'}`}>
                  {robResult.robustness_details.pass_dd ? 'PASS' : 'FAIL'} Max DD &le; 20% ({robResult.robustness_details.worst_dd}%)
                </div>
              </div>
            )}

            {/* Results Table */}
            <div className="rob-table-wrapper">
              <table className="trades-table rob-table">
                <thead>
                  <tr>
                    <th>Test Year</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>PF</th>
                    <th>Return</th>
                    <th>Max DD</th>
                    <th>Sharpe</th>
                    <th>Threshold</th>
                  </tr>
                </thead>
                <tbody>
                  {robResult.windows.map((w) => (
                    <tr key={w.label} className={w.error ? 'error-row' : (w.metrics && w.metrics.total_return_pct > 0 ? 'trade-win' : 'trade-loss')}>
                      <td className="year-cell">{w.label}</td>
                      {w.error ? (
                        <td colSpan={7} className="error-cell">{w.error}</td>
                      ) : w.metrics ? (
                        <>
                          <td>{w.metrics.total_trades}</td>
                          <td className={w.metrics.win_rate >= 55 ? 'positive' : w.metrics.win_rate >= 45 ? 'neutral' : 'negative'}>
                            {w.metrics.win_rate.toFixed(1)}%
                          </td>
                          <td className={w.metrics.profit_factor >= 1.8 ? 'positive' : w.metrics.profit_factor >= 1 ? 'neutral' : 'negative'}>
                            {w.metrics.profit_factor >= 999 ? '∞' : w.metrics.profit_factor.toFixed(2)}
                          </td>
                          <td className={w.metrics.total_return_pct >= 0 ? 'positive' : 'negative'}>
                            {w.metrics.total_return_pct >= 0 ? '+' : ''}{w.metrics.total_return_pct.toFixed(2)}%
                          </td>
                          <td className="negative">{w.metrics.max_drawdown_pct.toFixed(2)}%</td>
                          <td>{w.metrics.sharpe_ratio.toFixed(2)}</td>
                          <td>{w.threshold ? (w.threshold * 100).toFixed(0) + '%' : '-'}</td>
                        </>
                      ) : (
                        <td colSpan={7}>No data</td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {robResult && robResult.error && (
          <div className="backtest-error">{robResult.error}</div>
        )}
      </div>

      {/* Monte Carlo Section */}
      <div className="monte-carlo-section">
        <div className="mc-header">
          <h3>Monte Carlo Stress Test</h3>
          <button
            className="run-btn mc-btn"
            onClick={runMonteCarlo}
            disabled={mcLoading || selectedCoin === 'ALL'}
          >
            {mcLoading ? 'Running 100 sims...' : 'Run Monte Carlo'}
          </button>
        </div>

        {mcLoading && (
          <div className="backtest-loading">
            <div className="spinner"></div>
            <p>Running 100 randomized simulations...</p>
          </div>
        )}

        {mcResult && mcResult.summary && (
          <div className="mc-results">
            <div className={`mc-verdict ${mcResult.verdict === 'ROBUST' ? 'robust' : 'fragile'}`}>
              {mcResult.verdict === 'ROBUST' ? 'ROBUST' : 'FRAGILE'}
              <span className="mc-verdict-sub">
                {mcResult.summary.profitable_pct}% of {mcResult.simulations} simulations profitable
              </span>
            </div>

            <div className="mc-metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Profitable Sims</div>
                <div className={`metric-value ${mcResult.summary.profitable_pct >= 70 ? 'positive' : 'negative'}`}>
                  {mcResult.summary.profitable_pct}%
                </div>
                <div className="metric-sub">Target: 70%+</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Median Return</div>
                <div className={`metric-value ${mcResult.summary.median_return_pct >= 0 ? 'positive' : 'negative'}`}>
                  {mcResult.summary.median_return_pct >= 0 ? '+' : ''}{mcResult.summary.median_return_pct}%
                </div>
                <div className="metric-sub">50th percentile</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Worst Case</div>
                <div className="metric-value negative">
                  {mcResult.summary.worst_return_pct}%
                </div>
                <div className="metric-sub">Target: &gt; -10%</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Best Case</div>
                <div className="metric-value positive">
                  +{mcResult.summary.best_return_pct}%
                </div>
                <div className="metric-sub">Maximum upside</div>
              </div>
            </div>

            <div className="mc-distribution">
              <h4>Return Distribution</h4>
              <div className="percentile-bar">
                <div className="percentile-item">
                  <span className="p-label">5th</span>
                  <span className={mcResult.summary.percentile_5 >= 0 ? 'positive' : 'negative'}>
                    {mcResult.summary.percentile_5}%
                  </span>
                </div>
                <div className="percentile-item">
                  <span className="p-label">25th</span>
                  <span className={mcResult.summary.percentile_25 >= 0 ? 'positive' : 'negative'}>
                    {mcResult.summary.percentile_25}%
                  </span>
                </div>
                <div className="percentile-item">
                  <span className="p-label">Median</span>
                  <span className={mcResult.summary.median_return_pct >= 0 ? 'positive' : 'negative'}>
                    {mcResult.summary.median_return_pct}%
                  </span>
                </div>
                <div className="percentile-item">
                  <span className="p-label">75th</span>
                  <span className={mcResult.summary.percentile_75 >= 0 ? 'positive' : 'negative'}>
                    {mcResult.summary.percentile_75}%
                  </span>
                </div>
                <div className="percentile-item">
                  <span className="p-label">95th</span>
                  <span className={mcResult.summary.percentile_95 >= 0 ? 'positive' : 'negative'}>
                    {mcResult.summary.percentile_95}%
                  </span>
                </div>
              </div>

              {/* Pass/Fail Criteria */}
              <div className="mc-criteria">
                <div className={`criteria-item ${mcResult.pass_criteria.profitable_70pct ? 'pass' : 'fail'}`}>
                  {mcResult.pass_criteria.profitable_70pct ? 'PASS' : 'FAIL'} 70%+ simulations profitable
                </div>
                <div className={`criteria-item ${mcResult.pass_criteria.median_positive ? 'pass' : 'fail'}`}>
                  {mcResult.pass_criteria.median_positive ? 'PASS' : 'FAIL'} Median return positive
                </div>
                <div className={`criteria-item ${mcResult.pass_criteria.worst_case_above_neg10 ? 'pass' : 'fail'}`}>
                  {mcResult.pass_criteria.worst_case_above_neg10 ? 'PASS' : 'FAIL'} Worst case &gt; -10%
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Backtest;
