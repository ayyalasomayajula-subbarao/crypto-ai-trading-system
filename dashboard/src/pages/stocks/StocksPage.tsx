import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, Grid, Card, CardContent, Typography, Chip,
  TextField, Select, MenuItem, FormControl, InputLabel,
  CircularProgress, LinearProgress, Alert,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';

const STOCKS_API    = process.env.REACT_APP_STOCKS_API_URL || 'http://localhost:8000';
const STOCKS_WS_URL = process.env.REACT_APP_STOCKS_WS_URL  || 'ws://localhost:8000/ws/stocks/prices';

const VERDICT_COLORS: Record<string, string> = {
  STRONG_BUY: '#00e676', BUY: '#69f0ae', LEAN_BUY: '#b9f6ca',
  HOLD: '#ffd54f',
  LEAN_SELL: '#ffcc80', SELL: '#ff9100', STRONG_SELL: '#ff1744',
  AVOID: '#616161', AWAIT_DATA: '#455a64',
};

const StocksPage: React.FC = () => {
  const navigate = useNavigate();
  const [verdicts, setVerdicts] = useState<any[]>([]);
  const [prices, setPrices]     = useState<Record<string, any>>({});
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState('');
  const [search, setSearch]     = useState('');
  const [sectorFilter, setSector] = useState('All');
  const [verdictFilter, setVerdictF] = useState('All');
  const wsRef = useRef<WebSocket | null>(null);

  // WebSocket for live prices
  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>;
    const connect = () => {
      const ws = new WebSocket(STOCKS_WS_URL);
      ws.onclose = () => { reconnectTimer = setTimeout(connect, 3000); };
      ws.onerror = () => ws.close();
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'initial' || msg.type === 'prices') {
            setPrices(msg.data);
          }
        } catch {}
      };
      wsRef.current = ws;
    };
    connect();
    return () => { clearTimeout(reconnectTimer); wsRef.current?.close(); };
  }, []);

  // Scan fetch (verdicts change infrequently)
  useEffect(() => {
    const fetchScan = async () => {
      try {
        const sRes = await fetch(`${STOCKS_API}/stocks/scan`);
        setVerdicts(await sRes.json());
      } catch (e: any) {
        setError('Cannot connect to stocks API (port 8000)');
      } finally {
        setLoading(false);
      }
    };
    fetchScan();
    const interval = setInterval(fetchScan, 300_000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return (
    <Box display="flex" justifyContent="center" alignItems="center" height="60vh">
      <CircularProgress />
    </Box>
  );
  if (error) return <Box p={3}><Alert severity="warning">{error}</Alert></Box>;

  // Filters
  const sectors  = ['All', ...Array.from(new Set(verdicts.map(v => prices[v.symbol]?.sector).filter(Boolean)))];
  const vLabels  = ['All', 'STRONG_BUY', 'BUY', 'LEAN_BUY', 'HOLD', 'LEAN_SELL', 'SELL', 'STRONG_SELL', 'AVOID'];

  const filtered = verdicts.filter(v => {
    const p = prices[v.symbol];
    const matchSearch  = !search || v.symbol.includes(search.toUpperCase()) ||
                         v.display_name?.toLowerCase().includes(search.toLowerCase());
    const matchSector  = sectorFilter === 'All' || p?.sector === sectorFilter;
    const matchVerdict = verdictFilter === 'All' || v.verdict === verdictFilter;
    return matchSearch && matchSector && matchVerdict;
  });

  return (
    <Box p={3} maxWidth={1600}>
      <Typography variant="h4" fontWeight={700} gutterBottom>Stock Screener</Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        {verdicts.length} instruments · AI-scored · NSE F&O
      </Typography>

      {/* ── Filters ── */}
      <Box display="flex" gap={2} flexWrap="wrap" mb={3} mt={2}>
        <TextField
          size="small" placeholder="Search symbol or name..."
          value={search} onChange={e => setSearch(e.target.value)}
          InputProps={{ startAdornment: <SearchIcon sx={{ mr: 0.5, color: 'text.secondary' }} /> }}
          sx={{ minWidth: 220 }}
        />
        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel>Sector</InputLabel>
          <Select value={sectorFilter} onChange={e => setSector(e.target.value)} label="Sector">
            {sectors.map(s => <MenuItem key={s} value={s}>{s}</MenuItem>)}
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ minWidth: 160 }}>
          <InputLabel>Verdict</InputLabel>
          <Select value={verdictFilter} onChange={e => setVerdictF(e.target.value)} label="Verdict">
            {vLabels.map(v => (
              <MenuItem key={v} value={v}>
                <Chip size="small" label={v === 'All' ? v : v.replace('_',' ')}
                  sx={{ bgcolor: v === 'All' ? undefined : (VERDICT_COLORS[v] || '#616161') + '22',
                        color: v === 'All' ? 'text.primary' : VERDICT_COLORS[v] || '#9e9e9e' }} />
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <Typography variant="body2" color="text.secondary" sx={{ alignSelf: 'center' }}>
          {filtered.length} results
        </Typography>
      </Box>

      {/* ── Screener Grid ── */}
      <Grid container spacing={2}>
        {filtered.map(v => {
          const p   = prices[v.symbol];
          const col = VERDICT_COLORS[v.verdict] || '#616161';
          return (
            <Grid item xs={12} sm={6} md={4} lg={3} key={v.symbol}>
              <Card
                onClick={() => navigate(`/stocks/${v.symbol}`)}
                sx={{
                  bgcolor: 'rgba(255,255,255,0.03)',
                  border: `1px solid rgba(255,255,255,0.07)`,
                  borderRadius: 2,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  '&:hover': {
                    bgcolor: 'rgba(255,255,255,0.06)',
                    borderColor: col + '55',
                    transform: 'translateY(-2px)',
                    boxShadow: `0 4px 20px ${col}22`,
                  },
                }}>
                <CardContent sx={{ pb: '12px !important' }}>
                  <Box display="flex" justifyContent="space-between" alignItems="start" mb={1}>
                    <Box>
                      <Typography variant="body1" fontWeight={700}>{v.symbol}</Typography>
                      <Typography variant="caption" color="text.secondary" noWrap>
                        {v.display_name}
                      </Typography>
                    </Box>
                    <Chip size="small" label={v.verdict.replace(/_/g, ' ')}
                      sx={{ bgcolor: col + '22', color: col,
                            fontWeight: 600, fontSize: '0.6rem' }} />
                  </Box>

                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="h6" fontWeight={700}>
                      {p ? `₹${p.price.toLocaleString('en-IN', { maximumFractionDigits: 0 })}` : '—'}
                    </Typography>
                    {p && (
                      <Typography variant="body2"
                        sx={{ color: p.change_pct >= 0 ? '#69f0ae' : '#ff5252' }}>
                        {p.change_pct >= 0 ? '+' : ''}{p.change_pct.toFixed(2)}%
                      </Typography>
                    )}
                  </Box>

                  <LinearProgress variant="determinate" value={v.score}
                    sx={{ height: 4, borderRadius: 2, bgcolor: 'action.hover',
                          '& .MuiLinearProgress-bar': { bgcolor: col } }} />
                  <Box display="flex" justifyContent="space-between" mt={0.5}>
                    <Typography variant="caption" color="text.secondary">
                      Score {v.score.toFixed(0)}
                    </Typography>
                    {p?.sector && (
                      <Typography variant="caption" color="text.secondary">{p.sector}</Typography>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};

export default StocksPage;
