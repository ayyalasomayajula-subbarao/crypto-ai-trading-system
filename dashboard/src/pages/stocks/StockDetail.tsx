import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box, Grid, Card, CardContent, Typography, Chip, Tab, Tabs,
  CircularProgress, Button, Divider, LinearProgress, Alert,
  Table, TableBody, TableCell, TableHead, TableRow,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';
import {
  createChart, ColorType, CrosshairMode,
  CandlestickSeries, AreaSeries,
} from 'lightweight-charts';
import type { IChartApi, ISeriesApi, CandlestickData, AreaData, Time } from 'lightweight-charts';
import './StockDetail.css';

const STOCKS_API    = process.env.REACT_APP_STOCKS_API_URL || 'http://localhost:8000';
const STOCKS_WS_URL = process.env.REACT_APP_STOCKS_WS_URL  || 'ws://localhost:8000/ws/stocks/prices';

const VERDICT_COLORS: Record<string, string> = {
  STRONG_BUY: '#00e676', BUY: '#69f0ae', LEAN_BUY: '#b9f6ca',
  HOLD: '#ffd54f',
  LEAN_SELL: '#ffcc80', SELL: '#ff9100', STRONG_SELL: '#ff1744',
  AVOID: '#616161', AWAIT_DATA: '#455a64',
};

const SIGNAL_LABELS: Record<string, string> = {
  ml_model:          'ML Model',
  price_vs_sma21_1d: 'Price vs SMA-21',
  price_vs_sma50_1d: 'Price vs SMA-50',
  rsi_1d:            'RSI (Daily)',
  macd_1d:           'MACD',
  adx_1d:            'ADX Trend',
  pcr:               'Put/Call Ratio',
  india_vix:         'India VIX',
  fii_net:           'FII Flow',
  dii_net:           'DII Flow',
  oi_change:         'OI Change',
  delivery_pct:      'Delivery %',
  advance_decline:   'Advance/Decline',
  gift_nifty:        'GIFT Nifty Cue',
};

const CHART_TIMEFRAMES = [
  { label: '1H',  tf: '1h',  limit: 60   },
  { label: '1M',  tf: '1d',  limit: 22   },
  { label: '3M',  tf: '1d',  limit: 66   },
  { label: '6M',  tf: '1d',  limit: 132  },
  { label: '1Y',  tf: '1d',  limit: 252  },
  { label: '3Y',  tf: '1d',  limit: 756  },
  { label: 'ALL', tf: '1d',  limit: 2500 },
];

const TabPanel = ({ children, value, index }: any) => (
  <div hidden={value !== index} style={{ paddingTop: 16 }}>
    {value === index && children}
  </div>
);

// ── Lightweight-charts candlestick / area chart ───────────────────────────────

interface StockChartProps {
  symbol:       string;
  livePrice?:   number;
  verdictColor: string;
}

const StockChart: React.FC<StockChartProps> = ({ symbol, livePrice, verdictColor }) => {
  const containerRef  = useRef<HTMLDivElement>(null);
  const chartApiRef   = useRef<IChartApi | null>(null);
  const seriesRef     = useRef<ISeriesApi<'Candlestick'> | ISeriesApi<'Area'> | null>(null);
  const lastCandleRef = useRef<{ time: number; open: number; high: number; low: number; close: number } | null>(null);

  const [activeTf,  setActiveTf]  = useState(4);  // default 1Y
  const [chartType, setChartType] = useState<'area' | 'candlestick'>('area');
  const [loading,   setLoading]   = useState(false);
  const [priceInfo, setPriceInfo] = useState<{ price: number; changePct: number } | null>(null);

  const loadData = useCallback(async (tf: typeof CHART_TIMEFRAMES[number]) => {
    if (!chartApiRef.current || !seriesRef.current) return;
    setLoading(true);
    try {
      const res = await fetch(`${STOCKS_API}/stocks/klines/${symbol}?tf=${tf.tf}&limit=${tf.limit}`);
      if (!res.ok) return;
      const data   = await res.json();
      const candles: any[] = data.candles || [];
      if (!candles.length) return;

      const first = candles[0];
      const last  = candles[candles.length - 1];
      setPriceInfo({ price: last.close, changePct: ((last.close - first.open) / first.open) * 100 });
      lastCandleRef.current = { time: last.time, open: last.open, high: last.high, low: last.low, close: last.close };

      if (chartType === 'candlestick') {
        (seriesRef.current as ISeriesApi<'Candlestick'>).setData(
          candles.map(c => ({ time: c.time as Time, open: c.open, high: c.high, low: c.low, close: c.close }) as CandlestickData<Time>)
        );
      } else {
        (seriesRef.current as ISeriesApi<'Area'>).setData(
          candles.map(c => ({ time: c.time as Time, value: c.close }) as AreaData<Time>)
        );
      }
      chartApiRef.current.timeScale().fitContent();
    } finally {
      setLoading(false);
    }
  }, [symbol, chartType]);

  // Create/recreate chart on TF or chartType change
  useEffect(() => {
    if (!containerRef.current) return;
    if (chartApiRef.current) { chartApiRef.current.remove(); chartApiRef.current = null; seriesRef.current = null; }

    const chart = createChart(containerRef.current, {
      layout:  { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#888', fontSize: 11 },
      grid:    { vertLines: { color: '#1a1a2e' }, horzLines: { color: '#1a1a2e' } },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: '#444', width: 1, style: 3 as any, labelBackgroundColor: '#1a1a2e' },
        horzLine: { color: '#444', width: 1, style: 3 as any, labelBackgroundColor: '#1a1a2e' },
      },
      rightPriceScale: { borderColor: '#222' },
      timeScale: { borderColor: '#222', timeVisible: true },
      width:  containerRef.current.clientWidth,
      height: 380,
    });
    chartApiRef.current = chart;

    if (chartType === 'candlestick') {
      seriesRef.current = chart.addSeries(CandlestickSeries, {
        upColor: '#00e676', downColor: '#ff5252',
        borderUpColor: '#00e676', borderDownColor: '#ff5252',
        wickUpColor:   '#00e676', wickDownColor:   '#ff5252',
      });
    } else {
      seriesRef.current = chart.addSeries(AreaSeries, {
        lineColor:   verdictColor,
        topColor:    verdictColor + '40',
        bottomColor: verdictColor + '05',
        lineWidth:   2,
      });
    }

    loadData(CHART_TIMEFRAMES[activeTf]);

    const ro = new ResizeObserver(([entry]) => {
      chartApiRef.current?.applyOptions({ width: entry.contentRect.width });
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      if (chartApiRef.current) { chartApiRef.current.remove(); chartApiRef.current = null; seriesRef.current = null; }
    };
  }, [symbol, activeTf, chartType, verdictColor, loadData]);

  // Live price → update last candle in real-time
  useEffect(() => {
    if (!livePrice || !seriesRef.current || !lastCandleRef.current) return;
    const bar = lastCandleRef.current;
    const updated = { ...bar, close: livePrice, high: Math.max(bar.high, livePrice), low: Math.min(bar.low, livePrice) };
    lastCandleRef.current = updated;
    try {
      if (chartType === 'candlestick') {
        (seriesRef.current as ISeriesApi<'Candlestick'>).update(
          { time: updated.time as Time, open: updated.open, high: updated.high, low: updated.low, close: updated.close } as CandlestickData<Time>
        );
      } else {
        (seriesRef.current as ISeriesApi<'Area'>).update(
          { time: bar.time as Time, value: livePrice } as AreaData<Time>
        );
      }
      setPriceInfo(prev => prev ? { ...prev, price: livePrice } : null);
    } catch {}
  }, [livePrice, chartType]);

  return (
    <Box>
      {/* TF + type controls */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1.5} flexWrap="wrap" gap={1}>
        <Box display="flex" gap={0.5} flexWrap="wrap">
          {CHART_TIMEFRAMES.map((tf, i) => (
            <Button key={tf.label} size="small"
              variant={activeTf === i ? 'contained' : 'outlined'}
              onClick={() => setActiveTf(i)}
              sx={{ minWidth: 40, fontSize: '0.7rem', py: 0.3, px: 1 }}>
              {tf.label}
            </Button>
          ))}
        </Box>
        <Box display="flex" gap={0.5}>
          <Button size="small" variant={chartType === 'area' ? 'contained' : 'outlined'}
            onClick={() => setChartType('area')} sx={{ fontSize: '0.7rem', py: 0.3 }}>
            Area
          </Button>
          <Button size="small" variant={chartType === 'candlestick' ? 'contained' : 'outlined'}
            onClick={() => setChartType('candlestick')} sx={{ fontSize: '0.7rem', py: 0.3 }}>
            Candles
          </Button>
        </Box>
      </Box>

      {/* Price info */}
      {priceInfo && (
        <Box display="flex" gap={2} mb={1} alignItems="baseline">
          <Typography variant="h5" fontWeight={700}>
            ₹{priceInfo.price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </Typography>
          <Typography variant="body2"
            sx={{ color: priceInfo.changePct >= 0 ? '#69f0ae' : '#ff5252' }}>
            {priceInfo.changePct >= 0 ? '+' : ''}{priceInfo.changePct.toFixed(2)}% (period)
          </Typography>
        </Box>
      )}

      {/* Chart container */}
      <Box sx={{ position: 'relative' }}>
        {loading && (
          <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', zIndex: 1 }}>
            <CircularProgress size={28} />
          </Box>
        )}
        <div ref={containerRef} />
      </Box>
    </Box>
  );
};

// ── Main page ─────────────────────────────────────────────────────────────────

const StockDetail: React.FC = () => {
  const { symbol } = useParams<{ symbol: string }>();
  const navigate   = useNavigate();
  const [tab, setTab]         = useState(0);
  const [verdict, setVerdict] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [backtest, setBacktest] = useState<any>(null);
  const [loading, setLoading]  = useState(true);
  const [error, setError]      = useState('');
  const [livePrice, setLivePrice] = useState<number | undefined>();
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const sym = symbol?.toUpperCase() || '';

  // WebSocket — live price updates for this symbol
  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>;
    const connect = () => {
      const ws = new WebSocket(STOCKS_WS_URL);
      ws.onopen  = () => setWsConnected(true);
      ws.onclose = () => { setWsConnected(false); reconnectTimer = setTimeout(connect, 3000); };
      ws.onerror = () => ws.close();
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'initial' || msg.type === 'prices') {
            const p = msg.data?.[sym]?.price;
            if (p) setLivePrice(p);
          }
        } catch {}
      };
      wsRef.current = ws;
    };
    connect();
    return () => { clearTimeout(reconnectTimer); wsRef.current?.close(); };
  }, [sym]);

  const fetchVerdict = useCallback(async () => {
    try {
      const [vRes, hRes, bRes] = await Promise.all([
        fetch(`${STOCKS_API}/stocks/verdict/${sym}`),
        fetch(`${STOCKS_API}/stocks/verdict-history/${sym}?limit=30`),
        fetch(`${STOCKS_API}/stocks/backtest/${sym}`),
      ]);
      if (vRes.ok) setVerdict(await vRes.json());
      if (hRes.ok) setHistory(await hRes.json());
      if (bRes.ok) setBacktest(await bRes.json());
    } catch {
      setError('Cannot connect to stocks API (port 8000)');
    } finally {
      setLoading(false);
    }
  }, [sym]);

  useEffect(() => {
    fetchVerdict();
    const interval = setInterval(fetchVerdict, 300_000);
    return () => clearInterval(interval);
  }, [fetchVerdict]);

  if (loading) return (
    <Box display="flex" justifyContent="center" alignItems="center" height="60vh">
      <CircularProgress size={48} />
    </Box>
  );

  if (error || !verdict) return (
    <Box p={3}><Alert severity="warning">{error || 'Verdict not available'}</Alert></Box>
  );

  const color        = VERDICT_COLORS[verdict.verdict] || '#9e9e9e';
  const isBull       = verdict.direction === 'LONG';
  const isBear       = verdict.direction === 'SHORT';
  const wfTier       = verdict.wf_tier || 'NOT_VIABLE';
  const isNotViable  = wfTier === 'NOT_VIABLE';
  const wfTierColor  = wfTier === 'VIABLE' ? '#69f0ae' : wfTier === 'MARGINAL' ? '#ffd54f' : '#616161';
  const displayPrice = livePrice ?? verdict.current_price ?? 0;

  return (
    <Box className="stock-detail" p={3}>
      <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/stocks')}
        sx={{ mb: 2, color: 'text.secondary' }}>
        Back to Markets
      </Button>

      {/* ── Top Row ── */}
      <Grid container spacing={3} sx={{ mb: 3 }}>

        {/* Price Card */}
        <Grid item xs={12} md={4}>
          <Card className="detail-card">
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="start">
                <Typography variant="body2" color="text.secondary">{verdict.display_name}</Typography>
                <Chip size="small"
                  icon={<FiberManualRecordIcon sx={{ fontSize: '0.6rem !important' }} />}
                  label={wsConnected ? 'LIVE' : '—'}
                  color={wsConnected ? 'success' : 'default'}
                  variant="outlined"
                  sx={{ fontSize: '0.65rem', height: 20 }}
                />
              </Box>
              <Typography variant="h3" fontWeight={800}>
                ₹{displayPrice.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </Typography>
              <Box display="flex" gap={1} mt={1}>
                <Chip label={verdict.verdict}
                  sx={{ bgcolor: color + '22', color, fontWeight: 700, fontSize: '0.85rem' }} />
                {verdict.direction !== 'NEUTRAL' && (
                  <Chip label={verdict.direction}
                    icon={isBull ? <TrendingUpIcon /> : <TrendingDownIcon />}
                    color={isBull ? 'success' : 'error'}
                    variant="outlined" size="small" />
                )}
              </Box>
              <Box mt={2}>
                <LinearProgress variant="determinate" value={verdict.score}
                  sx={{ height: 8, borderRadius: 4, bgcolor: 'action.hover',
                        '& .MuiLinearProgress-bar': { bgcolor: color } }} />
                <Typography variant="body2" color="text.secondary" mt={0.5}>
                  Conviction: {verdict.score.toFixed(0)} / 100
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* TP / SL */}
        <Grid item xs={12} md={4}>
          <Card className="detail-card">
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>Trade Setup</Typography>
              <Grid container spacing={1}>
                {[
                  { label: 'Entry',                    val: verdict.entry_price,  color: '#90caf9' },
                  { label: `TP (+${verdict.tp_pct}%)`, val: verdict.target_price, color: '#69f0ae' },
                  { label: `SL (-${verdict.sl_pct}%)`, val: verdict.sl_price,     color: '#ff5252' },
                ].map(({ label, val, color: c }) => (
                  <Grid item xs={4} key={label}>
                    <Box textAlign="center" p={1}
                      sx={{ bgcolor: c + '11', borderRadius: 2, border: `1px solid ${c}33` }}>
                      <Typography variant="caption" sx={{ color: c }}>{label}</Typography>
                      <Typography variant="body2" fontWeight={700}>
                        ₹{val?.toLocaleString('en-IN', { maximumFractionDigits: 2 }) || '—'}
                      </Typography>
                    </Box>
                  </Grid>
                ))}
              </Grid>
              <Divider sx={{ my: 1.5 }} />
              {/* ML Bias — shows model direction clearly, flags conflict with overall verdict */}
              {(() => {
                const pUp   = (verdict.ml_p_up   || 0) * 100;
                const pDown = (verdict.ml_p_down  || 0) * 100;
                const pSide = Math.max(0, 100 - pUp - pDown);
                const mlBull = pUp > pDown && pUp > pSide;
                const mlBear = pDown > pUp && pDown > pSide;
                const mlLabel = mlBull ? 'Bullish' : mlBear ? 'Bearish' : 'Neutral';
                const mlColor = mlBull ? '#69f0ae' : mlBear ? '#ff5252' : '#ffd54f';
                const verdictBull = ['STRONG_BUY','BUY','LEAN_BUY'].includes(verdict.verdict);
                const verdictBear = ['STRONG_SELL','SELL','LEAN_SELL'].includes(verdict.verdict);
                const conflict = (mlBear && verdictBull) || (mlBull && verdictBear);
                return (
                  <Box>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Box>
                        <Typography variant="caption" color="text.secondary">ML Bias</Typography>
                        <Box display="flex" alignItems="center" gap={0.5}>
                          <Typography variant="body2" fontWeight={700} sx={{ color: mlColor }}>
                            {mlLabel}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            ({pUp.toFixed(0)}% ↑ · {pDown.toFixed(0)}% ↓ · {pSide.toFixed(0)}% →)
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                    {conflict && (
                      <Typography variant="caption" sx={{ color: '#ffa726', display: 'block', mt: 0.5 }}>
                        ⚠ ML leans {mlLabel.toLowerCase()} — technical signals override
                      </Typography>
                    )}
                  </Box>
                );
              })()}
            </CardContent>
          </Card>
        </Grid>

        {/* Actions */}
        <Grid item xs={12} md={4}>
          <Card className="detail-card" sx={{ height: '100%' }}>
            <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="body2" color="text.secondary">Actions</Typography>
                <Chip size="small" label={wfTier}
                  sx={{ fontSize: '0.65rem', height: 20,
                        bgcolor: wfTierColor + '22', color: wfTierColor, fontWeight: 700 }} />
              </Box>
              {isNotViable && (
                <Typography variant="caption" sx={{ color: '#616161' }}>
                  WF backtest not viable — trading disabled for this symbol
                </Typography>
              )}
              <Button fullWidth variant="contained"
                color={isBull ? 'success' : isBear ? 'error' : 'primary'}
                disabled={verdict.direction === 'NEUTRAL' || isNotViable}
                onClick={async () => {
                  await fetch(`${STOCKS_API}/stocks/paper-trading/open`,
                    { method: 'POST', headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ symbol: sym }) });
                  alert(`Paper trade opened: ${verdict.direction} ${sym}`);
                }}
                sx={{ fontWeight: 700 }}>
                {isNotViable ? 'Trading Disabled' : verdict.direction === 'NEUTRAL' ? 'No Signal' : `Paper ${verdict.direction} ${sym}`}
              </Button>
              <Button fullWidth variant="outlined" color="secondary"
                onClick={() => navigate('/stocks/market/pulse')}>
                View Market Pulse
              </Button>
              <Button fullWidth variant="outlined"
                onClick={() => window.open(
                  `https://www.nseindia.com/get-quotes/equity?symbol=${sym}`, '_blank'
                )}>
                NSE Quote
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* ── Tabs ── */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tab} onChange={(_, v) => setTab(v)}>
          <Tab label="Signals" />
          <Tab label="Chart" />
          <Tab label="Backtest" />
          <Tab label="History" />
        </Tabs>
      </Box>

      {/* Signals */}
      <TabPanel value={tab} index={0}>
        <Grid container spacing={2}>
          {(verdict.signals || []).map((sig: any) => {
            const pct      = Math.round(((sig.value + 1) / 2) * 100);
            const sigColor = sig.value > 0.1 ? '#69f0ae' : sig.value < -0.1 ? '#ff5252' : '#ffd54f';
            return (
              <Grid item xs={12} sm={6} md={4} key={sig.name}>
                <Card sx={{ bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 2 }}>
                  <CardContent sx={{ py: 1.5 }}>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="body2" fontWeight={500}>
                        {SIGNAL_LABELS[sig.name] || sig.name}
                      </Typography>
                      <Chip size="small"
                        label={sig.bullish ? 'BULL' : sig.bearish ? 'BEAR' : 'NEUTRAL'}
                        sx={{ bgcolor: sigColor + '22', color: sigColor, fontSize: '0.65rem' }} />
                    </Box>
                    <LinearProgress variant="determinate" value={Math.max(0, Math.min(100, pct))}
                      sx={{ mt: 1, height: 4, borderRadius: 2, bgcolor: 'action.hover',
                            '& .MuiLinearProgress-bar': { bgcolor: sigColor } }} />
                    <Box display="flex" justifyContent="space-between" mt={0.5}>
                      <Typography variant="caption" color="text.secondary">Weight: {sig.weight}x</Typography>
                      <Typography variant="caption" sx={{ color: sigColor }}>
                        {sig.value > 0 ? '+' : ''}{(sig.value * 100).toFixed(1)}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      </TabPanel>

      {/* Chart — lightweight-charts with live WS price updates */}
      <TabPanel value={tab} index={1}>
        <Card sx={{ bgcolor: 'rgba(255,255,255,0.02)', p: 2, borderRadius: 2 }}>
          <StockChart symbol={sym} livePrice={livePrice} verdictColor={color} />
        </Card>
      </TabPanel>

      {/* Backtest */}
      <TabPanel value={tab} index={2}>
        {backtest ? (
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Box display="flex" gap={2} flexWrap="wrap" mb={2}>
                {[
                  { label: 'Verdict',    val: backtest.verdict,
                    color: backtest.verdict === 'VIABLE' ? '#69f0ae' : backtest.verdict === 'MARGINAL' ? '#ffd54f' : '#ff5252' },
                  { label: 'Avg Sharpe', val: backtest.avg_sharpe?.toFixed(3),
                    color: backtest.avg_sharpe > 0 ? '#69f0ae' : '#ff5252' },
                  { label: 'Avg WR',     val: `${(backtest.avg_wr * 100).toFixed(1)}%`, color: '#90caf9' },
                  { label: 'Trades',     val: backtest.total_trades,  color: '#ce93d8' },
                  { label: 'Threshold',  val: backtest.best_threshold, color: '#ffd54f' },
                ].map(({ label, val, color: c }) => (
                  <Box key={label} sx={{ bgcolor: c + '11', border: `1px solid ${c}33`, borderRadius: 2, px: 2, py: 1 }}>
                    <Typography variant="caption" color="text.secondary">{label}</Typography>
                    <Typography variant="body1" fontWeight={700} sx={{ color: c }}>{val}</Typography>
                  </Box>
                ))}
              </Box>
            </Grid>
            {backtest.fold_results?.length > 0 && (
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary" gutterBottom>Fold Results</Typography>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Fold</TableCell><TableCell>Sharpe</TableCell>
                      <TableCell>Win Rate</TableCell><TableCell>Trades</TableCell>
                      <TableCell>Long</TableCell><TableCell>Short</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {backtest.fold_results.map((f: any) => (
                      <TableRow key={f.fold}>
                        <TableCell>Fold {f.fold}</TableCell>
                        <TableCell sx={{ color: f.sharpe > 0 ? '#69f0ae' : '#ff5252' }}>{f.sharpe?.toFixed(3)}</TableCell>
                        <TableCell>{(f.wr * 100).toFixed(1)}%</TableCell>
                        <TableCell>{f.n_trades}</TableCell>
                        <TableCell sx={{ color: '#69f0ae' }}>{f.long_trades}</TableCell>
                        <TableCell sx={{ color: '#ff5252' }}>{f.short_trades}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Grid>
            )}
          </Grid>
        ) : (
          <Alert severity="info">
            No backtest results yet. Run: <code>python walk_forward.py --symbol {sym}</code>
          </Alert>
        )}
      </TabPanel>

      {/* History */}
      <TabPanel value={tab} index={3}>
        {history.length > 0 ? (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Date</TableCell><TableCell>Verdict</TableCell>
                <TableCell>Score</TableCell><TableCell>Direction</TableCell>
                <TableCell>Entry</TableCell><TableCell>Target</TableCell><TableCell>SL</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {history.map((h: any, i: number) => (
                <TableRow key={i}>
                  <TableCell sx={{ fontSize: '0.75rem' }}>
                    {new Date(h.timestamp).toLocaleDateString('en-IN')}
                  </TableCell>
                  <TableCell>
                    <Chip size="small" label={h.verdict}
                      sx={{ bgcolor: (VERDICT_COLORS[h.verdict] || '#616161') + '22',
                            color: VERDICT_COLORS[h.verdict] || '#9e9e9e', fontSize: '0.65rem' }} />
                  </TableCell>
                  <TableCell>{h.score?.toFixed(1)}</TableCell>
                  <TableCell sx={{ color: h.direction === 'LONG' ? '#69f0ae' : h.direction === 'SHORT' ? '#ff5252' : '#9e9e9e' }}>
                    {h.direction || '—'}
                  </TableCell>
                  <TableCell>₹{h.entry_price?.toFixed(2)}</TableCell>
                  <TableCell sx={{ color: '#69f0ae' }}>₹{h.target_price?.toFixed(2)}</TableCell>
                  <TableCell sx={{ color: '#ff5252' }}>₹{h.sl_price?.toFixed(2)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <Typography color="text.secondary">No verdict history yet.</Typography>
        )}
      </TabPanel>
    </Box>
  );
};

export default StockDetail;
