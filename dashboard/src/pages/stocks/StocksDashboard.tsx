import React, { useState, useEffect, useRef } from 'react';
import {
  Box, Grid, Card, CardContent, Typography, Chip,
  CircularProgress, Divider, LinearProgress, Alert,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import { useNavigate } from 'react-router-dom';
import './StocksDashboard.css';

const STOCKS_API    = process.env.REACT_APP_STOCKS_API_URL || 'http://localhost:8000';
const STOCKS_WS_URL = process.env.REACT_APP_STOCKS_WS_URL  || 'ws://localhost:8000/ws/stocks/prices';

interface MarketOverview {
  india_vix: { value: number; change_pct: number; level: string } | null;
  fii_dii: {
    fii_net: number; dii_net: number;
    fii_7d_cumulative: number; dii_7d_cumulative: number;
    fii_net_label: string; dii_net_label: string;
  } | null;
  option_chain: {
    NIFTY50?: { pcr: number; pcr_7d_avg: number; max_pain: number; iv_percentile: number };
    BANKNIFTY?: { pcr: number; pcr_7d_avg: number; max_pain: number; iv_percentile: number };
  };
  market_open: boolean;
}

interface StockPrice {
  symbol: string; display_name: string; price: number;
  change_pct: number; sector: string; type: string;
}

const VIX_COLORS: Record<string, string> = {
  EXTREME_FEAR: '#ff1744',
  HIGH_FEAR: '#ff5722',
  NEUTRAL: '#ffd54f',
  COMPLACENCY: '#69f0ae',
};

const PCR_LABEL = (pcr: number) => {
  if (pcr < 0.7)  return { label: 'Strong Bullish', color: '#00e676' };
  if (pcr < 0.85) return { label: 'Bullish',         color: '#69f0ae' };
  if (pcr < 1.1)  return { label: 'Neutral',          color: '#ffd54f' };
  if (pcr < 1.3)  return { label: 'Bearish',           color: '#ff9100' };
  return              { label: 'Strong Bearish',   color: '#ff1744' };
};

const VERDICT_COLORS: Record<string, string> = {
  STRONG_BUY: '#00e676', BUY: '#69f0ae', LEAN_BUY: '#b9f6ca',
  HOLD: '#ffd54f',
  LEAN_SELL: '#ffcc80', SELL: '#ff9100', STRONG_SELL: '#ff1744',
  AVOID: '#616161', AWAIT_DATA: '#455a64',
};

const StocksDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [overview, setOverview] = useState<MarketOverview | null>(null);
  const [prices, setPrices] = useState<Record<string, StockPrice>>({});
  const [verdicts, setVerdicts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);
  const [error, setError] = useState('');
  const wsRef = useRef<WebSocket | null>(null);

  // WebSocket for live prices
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
            setPrices(msg.data);
            setLoading(false);
          }
        } catch {}
      };
      wsRef.current = ws;
    };
    connect();
    return () => {
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, []);

  // Overview + scan via polling (not price-sensitive)
  useEffect(() => {
    const fetchOverviewAndScan = async () => {
      try {
        const [ovRes, scanRes] = await Promise.all([
          fetch(`${STOCKS_API}/stocks/market/overview`),
          fetch(`${STOCKS_API}/stocks/scan`),
        ]);
        if (ovRes.ok)   setOverview(await ovRes.json());
        if (scanRes.ok) setVerdicts(await scanRes.json());
      } catch (e: any) {
        setError(`API error: ${e.message || 'Cannot connect to stocks API (port 8000)'}`);
        setLoading(false);
      }
    };
    fetchOverviewAndScan();
    const interval = setInterval(fetchOverviewAndScan, 300_000); // every 5 min
    return () => clearInterval(interval);
  }, []);

  if (loading) return (
    <Box display="flex" justifyContent="center" alignItems="center" height="60vh">
      <CircularProgress size={48} />
    </Box>
  );

  if (error) return (
    <Box p={3}>
      <Alert severity="warning" sx={{ mb: 2 }}>{error}</Alert>
      <Typography variant="body2" color="text.secondary">
        Run: <code>uvicorn main:app --port 8000 --reload</code>
      </Typography>
    </Box>
  );

  const indexSymbols  = verdicts.filter(v => ['NIFTY50','BANKNIFTY','NIFTYIT'].includes(v.symbol));
  const stockVerdicts = verdicts.filter(v => !['NIFTY50','BANKNIFTY','NIFTYIT'].includes(v.symbol));

  return (
    <Box className="stocks-dashboard">
      {/* ── Header ── */}
      <Box className="stocks-header">
        <Box>
          <Typography variant="h4" fontWeight={700}>Indian Markets</Typography>
          <Typography variant="body2" color="text.secondary">
            NSE · BSE · F&O Analysis
            <Chip
              size="small"
              label={overview?.market_open ? 'MARKET OPEN' : 'MARKET CLOSED'}
              color={overview?.market_open ? 'success' : 'default'}
              sx={{ ml: 1 }}
            />
            <Chip
              size="small"
              label={wsConnected ? 'LIVE' : 'CONNECTING…'}
              color={wsConnected ? 'success' : 'warning'}
              variant="outlined"
              sx={{ ml: 0.5 }}
            />
          </Typography>
        </Box>
      </Box>

      {/* ── Market Pulse Row ── */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {/* India VIX */}
        <Grid item xs={12} sm={6} md={3}>
          <Card className="pulse-card vix-card">
            <CardContent>
              <Typography variant="caption" color="text.secondary">INDIA VIX</Typography>
              {overview?.india_vix ? (
                <>
                  <Typography variant="h3" fontWeight={700}
                    sx={{ color: VIX_COLORS[overview.india_vix.level] || '#ffd54f' }}>
                    {overview.india_vix.value.toFixed(1)}
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1}>
                    {overview.india_vix.change_pct >= 0
                      ? <TrendingUpIcon fontSize="small" color="error" />
                      : <TrendingDownIcon fontSize="small" color="success" />}
                    <Typography variant="body2">
                      {overview.india_vix.change_pct > 0 ? '+' : ''}
                      {overview.india_vix.change_pct.toFixed(2)}%
                    </Typography>
                    <Chip size="small" label={overview.india_vix.level.replace('_', ' ')}
                      sx={{ bgcolor: VIX_COLORS[overview.india_vix.level] + '22',
                            color: VIX_COLORS[overview.india_vix.level] }} />
                  </Box>
                </>
              ) : <Typography color="text.secondary">—</Typography>}
            </CardContent>
          </Card>
        </Grid>

        {/* FII Flow */}
        <Grid item xs={12} sm={6} md={3}>
          <Card className="pulse-card">
            <CardContent>
              <Typography variant="caption" color="text.secondary">FII NET (₹ Cr)</Typography>
              {overview?.fii_dii ? (
                <>
                  <Typography variant="h4" fontWeight={700}
                    sx={{ color: overview.fii_dii.fii_net >= 0 ? '#69f0ae' : '#ff5252' }}>
                    {overview.fii_dii.fii_net >= 0 ? '+' : ''}
                    {overview.fii_dii.fii_net.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    7d: {overview.fii_dii.fii_7d_cumulative >= 0 ? '+' : ''}
                    {overview.fii_dii.fii_7d_cumulative.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </Typography>
                  <Chip size="small" label={`FII ${overview.fii_dii.fii_net_label}`}
                    color={overview.fii_dii.fii_net >= 0 ? 'success' : 'error'} sx={{ mt: 0.5 }} />
                </>
              ) : <Typography color="text.secondary">—</Typography>}
            </CardContent>
          </Card>
        </Grid>

        {/* DII Flow */}
        <Grid item xs={12} sm={6} md={3}>
          <Card className="pulse-card">
            <CardContent>
              <Typography variant="caption" color="text.secondary">DII NET (₹ Cr)</Typography>
              {overview?.fii_dii ? (
                <>
                  <Typography variant="h4" fontWeight={700}
                    sx={{ color: overview.fii_dii.dii_net >= 0 ? '#69f0ae' : '#ff5252' }}>
                    {overview.fii_dii.dii_net >= 0 ? '+' : ''}
                    {overview.fii_dii.dii_net.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    7d: {overview.fii_dii.dii_7d_cumulative >= 0 ? '+' : ''}
                    {overview.fii_dii.dii_7d_cumulative.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </Typography>
                  <Chip size="small" label={`DII ${overview.fii_dii.dii_net_label}`}
                    color={overview.fii_dii.dii_net >= 0 ? 'success' : 'error'} sx={{ mt: 0.5 }} />
                </>
              ) : <Typography color="text.secondary">—</Typography>}
            </CardContent>
          </Card>
        </Grid>

        {/* NIFTY PCR */}
        <Grid item xs={12} sm={6} md={3}>
          <Card className="pulse-card">
            <CardContent>
              <Typography variant="caption" color="text.secondary">NIFTY PCR</Typography>
              {overview?.option_chain?.NIFTY50 ? (
                <>
                  <Typography variant="h3" fontWeight={700}
                    sx={{ color: PCR_LABEL(overview.option_chain.NIFTY50.pcr_7d_avg || overview.option_chain.NIFTY50.pcr).color }}>
                    {(overview.option_chain.NIFTY50.pcr_7d_avg || overview.option_chain.NIFTY50.pcr).toFixed(2)}
                  </Typography>
                  <Chip size="small"
                    label={PCR_LABEL(overview.option_chain.NIFTY50.pcr_7d_avg || overview.option_chain.NIFTY50.pcr).label}
                    sx={{
                      bgcolor: PCR_LABEL(overview.option_chain.NIFTY50.pcr).color + '22',
                      color: PCR_LABEL(overview.option_chain.NIFTY50.pcr).color,
                      mt: 0.5,
                    }} />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                    Max Pain: ₹{overview.option_chain.NIFTY50.max_pain?.toFixed(0)}
                  </Typography>
                </>
              ) : <Typography color="text.secondary">—</Typography>}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* ── Index Cards ── */}
      <Typography variant="h6" fontWeight={600} sx={{ mb: 1.5 }}>Indices</Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {indexSymbols.map(v => {
          const p = prices[v.symbol];
          return (
            <Grid item xs={12} sm={6} md={4} key={v.symbol}>
              <Card className="instrument-card" onClick={() => navigate(`/stocks/${v.symbol}`)}
                sx={{ cursor: 'pointer', '&:hover': { borderColor: 'primary.main' }, border: '1px solid transparent' }}>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="start">
                    <Box>
                      <Typography variant="body2" color="text.secondary">{v.display_name}</Typography>
                      <Typography variant="h5" fontWeight={700}>
                        ₹{p?.price.toLocaleString('en-IN', { maximumFractionDigits: 0 }) || '—'}
                      </Typography>
                      {p && (
                        <Typography variant="body2"
                          sx={{ color: p.change_pct >= 0 ? '#69f0ae' : '#ff5252' }}>
                          {p.change_pct >= 0 ? '+' : ''}{p.change_pct.toFixed(2)}%
                        </Typography>
                      )}
                    </Box>
                    <Chip size="small" label={v.verdict}
                      sx={{ bgcolor: (VERDICT_COLORS[v.verdict] || '#616161') + '22',
                            color: VERDICT_COLORS[v.verdict] || '#9e9e9e',
                            fontWeight: 600 }} />
                  </Box>
                  <LinearProgress variant="determinate" value={v.score}
                    sx={{ mt: 1.5, height: 4, borderRadius: 2,
                          bgcolor: 'action.hover',
                          '& .MuiLinearProgress-bar': {
                            bgcolor: VERDICT_COLORS[v.verdict] || '#616161' } }} />
                  <Typography variant="caption" color="text.secondary">
                    Score: {v.score.toFixed(0)}/100
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* ── Top Stock Signals ── */}
      <Typography variant="h6" fontWeight={600} sx={{ mb: 1.5 }}>
        Top Signals — F&O Stocks
      </Typography>
      <Grid container spacing={2}>
        {stockVerdicts.slice(0, 12).map(v => {
          const p = prices[v.symbol];
          return (
            <Grid item xs={12} sm={6} md={4} lg={3} key={v.symbol}>
              <Card className="stock-signal-card"
                onClick={() => navigate(`/stocks/${v.symbol}`)}
                sx={{ cursor: 'pointer', '&:hover': { borderColor: 'primary.main' },
                      border: '1px solid transparent' }}>
                <CardContent sx={{ py: 1.5 }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Box>
                      <Typography variant="body2" fontWeight={600}>{v.symbol}</Typography>
                      <Typography variant="caption" color="text.secondary">{v.display_name}</Typography>
                    </Box>
                    <Box textAlign="right">
                      <Chip size="small" label={v.verdict}
                        sx={{ bgcolor: (VERDICT_COLORS[v.verdict] || '#616161') + '22',
                              color: VERDICT_COLORS[v.verdict] || '#9e9e9e',
                              fontWeight: 600, fontSize: '0.65rem' }} />
                      {p && (
                        <Typography variant="caption" display="block"
                          sx={{ color: p.change_pct >= 0 ? '#69f0ae' : '#ff5252' }}>
                          {p.change_pct >= 0 ? '+' : ''}{p.change_pct.toFixed(2)}%
                        </Typography>
                      )}
                    </Box>
                  </Box>
                  <LinearProgress variant="determinate" value={v.score}
                    sx={{ mt: 1, height: 3, borderRadius: 2,
                          bgcolor: 'action.hover',
                          '& .MuiLinearProgress-bar': {
                            bgcolor: VERDICT_COLORS[v.verdict] || '#616161' } }} />
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};

export default StocksDashboard;
