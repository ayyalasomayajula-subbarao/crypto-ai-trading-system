import React, { useState, useEffect } from 'react';
import {
  Box, Grid, Card, CardContent, Typography, Chip, Button,
  CircularProgress, Alert, Table, TableBody, TableCell,
  TableHead, TableRow, Divider,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import RadarIcon from '@mui/icons-material/Radar';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';

const STOCKS_API = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const EVENT_COLORS: Record<string, string> = {
  SCAN_START: '#546e7a',
  SIGNAL:     '#29b6f6',
  ENTRY:      '#66bb6a',
  SKIP:       '#78909c',
  EXIT:       '#ffa726',
  ERROR:      '#ef5350',
};

const StocksPortfolio: React.FC = () => {
  const [status, setStatus]     = useState<any>(null);
  const [trades, setTrades]     = useState<any>({ open: [], closed: [] });
  const [activity, setActivity] = useState<any>(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState('');

  const refresh = async () => {
    try {
      const [sRes, tRes, aRes] = await Promise.all([
        fetch(`${STOCKS_API}/stocks/paper-trading/status`),
        fetch(`${STOCKS_API}/stocks/paper-trading/trades`),
        fetch(`${STOCKS_API}/stocks/paper-trading/activity?limit=30`),
      ]);
      setStatus(await sRes.json());
      setTrades(await tRes.json());
      setActivity(await aRes.json());
    } catch (e: any) {
      setError('Cannot connect to stocks API');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, []);

  const closePos = async (symbol: string) => {
    await fetch(`${STOCKS_API}/stocks/paper-trading/close/${symbol}`, { method: 'POST' });
    refresh();
  };

  if (loading) return (
    <Box display="flex" justifyContent="center" alignItems="center" height="60vh">
      <CircularProgress />
    </Box>
  );
  if (error) return <Box p={3}><Alert severity="warning">{error}</Alert></Box>;

  const m = status?.metrics || {};
  const isRunning = status?.running ?? false;
  const marketOpen = status?.market_open ?? false;

  return (
    <Box p={3} maxWidth={1400}>
      {/* ── Header ── */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" fontWeight={700}>Stocks Portfolio</Typography>
          <Box display="flex" alignItems="center" gap={1} mt={0.5}>
            <FiberManualRecordIcon
              sx={{ fontSize: 10, color: isRunning ? '#66bb6a' : '#546e7a',
                    animation: isRunning && marketOpen ? 'pulse 1.5s infinite' : 'none',
                    '@keyframes pulse': { '0%,100%': { opacity: 1 }, '50%': { opacity: 0.3 } } }} />
            <Typography variant="body2" color="text.secondary">
              {isRunning ? 'Paper trading active' : 'Paper trading stopped'}
            </Typography>
            <Chip size="small"
              label={marketOpen ? 'Market Open' : 'Market Closed'}
              color={marketOpen ? 'success' : 'default'} />
            {isRunning && !marketOpen && (
              <Chip size="small" label="Scans at 09:15 IST" variant="outlined" sx={{ fontSize: '0.7rem' }} />
            )}
          </Box>
        </Box>
        <Box display="flex" gap={1}>
          <Button size="small" variant="outlined"
            onClick={() => { fetch(`${STOCKS_API}/stocks/paper-trading/start`, { method: 'POST' }); refresh(); }}>
            Start
          </Button>
          <Button size="small" variant="outlined" color="warning"
            onClick={() => { fetch(`${STOCKS_API}/stocks/paper-trading/stop`, { method: 'POST' }); refresh(); }}>
            Stop
          </Button>
          <Button size="small" variant="outlined" color="error"
            onClick={async () => {
              if (window.confirm('Reset all paper trades?')) {
                await fetch(`${STOCKS_API}/stocks/paper-trading/reset`, { method: 'POST' });
                refresh();
              }
            }}>
            Reset
          </Button>
        </Box>
      </Box>

      {/* ── Metric Cards ── */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { label: 'Total P&L', val: `₹${(m.total_pnl || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`, sub: `${m.total_pnl_pct || 0}%`, color: (m.total_pnl || 0) >= 0 ? '#69f0ae' : '#ff5252' },
          { label: 'Win Rate', val: `${((m.win_rate || 0) * 100).toFixed(1)}%`, sub: `${m.total_trades || 0} trades`, color: '#90caf9' },
          { label: 'Sharpe', val: (m.sharpe || 0).toFixed(3), sub: 'annualised', color: m.sharpe > 0 ? '#69f0ae' : '#ff5252' },
          { label: 'Profit Factor', val: (m.profit_factor || 0).toFixed(2), sub: 'wins/losses', color: '#ce93d8' },
          { label: 'Open Positions', val: m.open_positions || 0, sub: `of ${5} max`, color: '#ffd54f' },
          { label: 'Daily P&L', val: `₹${(m.daily_pnl || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`, sub: 'today', color: (m.daily_pnl || 0) >= 0 ? '#69f0ae' : '#ff5252' },
        ].map(({ label, val, sub, color }) => (
          <Grid item xs={6} sm={4} md={2} key={label}>
            <Card sx={{ bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 2 }}>
              <CardContent sx={{ py: 1.5, '&:last-child': { pb: '12px' } }}>
                <Typography variant="caption" color="text.secondary">{label}</Typography>
                <Typography variant="h6" fontWeight={700} sx={{ color }}>{val}</Typography>
                <Typography variant="caption" color="text.secondary">{sub}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* ── Open Positions ── */}
      <Typography variant="h6" fontWeight={600} gutterBottom>Open Positions</Typography>
      {trades.open?.length === 0 ? (
        <Card sx={{ bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2, p: 2, mb: 3 }}>
          <Typography color="text.secondary" textAlign="center">No open positions</Typography>
        </Card>
      ) : (
        <Box sx={{ overflowX: 'auto', mb: 3 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Symbol</TableCell>
                <TableCell>Direction</TableCell>
                <TableCell>Entry</TableCell>
                <TableCell>Qty</TableCell>
                <TableCell>Target</TableCell>
                <TableCell>SL</TableCell>
                <TableCell align="right">Unrealised P&L</TableCell>
                <TableCell>Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {trades.open.map((pos: any) => (
                <TableRow key={pos.symbol}>
                  <TableCell sx={{ fontWeight: 'bold' }}>{pos.symbol}</TableCell>
                  <TableCell>
                    <Chip size="small"
                      label={pos.direction}
                      color={pos.direction === 'LONG' ? 'success' : 'error'}
                      icon={pos.direction === 'LONG' ? <TrendingUpIcon /> : <TrendingDownIcon />} />
                  </TableCell>
                  <TableCell>₹{pos.entry_price?.toFixed(2)}</TableCell>
                  <TableCell>{pos.qty} ({pos.lots} lot)</TableCell>
                  <TableCell sx={{ color: '#69f0ae' }}>₹{pos.target_price?.toFixed(2)}</TableCell>
                  <TableCell sx={{ color: '#ff5252' }}>₹{pos.sl_price?.toFixed(2)}</TableCell>
                  <TableCell align="right"
                    sx={{ color: (pos.unrealized_pnl || 0) >= 0 ? '#69f0ae' : '#ff5252', fontWeight: 600 }}>
                    ₹{(pos.unrealized_pnl || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                    &nbsp;({pos.unrealized_pct || 0}%)
                  </TableCell>
                  <TableCell>
                    <Button size="small" variant="outlined" color="error"
                      onClick={() => closePos(pos.symbol)}>
                      Close
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      )}

      {/* ── Scanner Activity Log ── */}
      <Box display="flex" alignItems="center" gap={1} mb={1}>
        <RadarIcon sx={{ fontSize: 18, color: '#90caf9' }} />
        <Typography variant="h6" fontWeight={600}>Scanner Activity</Typography>
        <Typography variant="caption" color="text.secondary">
          — auto-scans every 15 min · {activity?.next_scan || ''}
        </Typography>
      </Box>
      <Card sx={{ bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2, mb: 3, maxHeight: 280, overflow: 'auto' }}>
        {!activity?.events?.length ? (
          <Box p={2}>
            <Typography color="text.secondary" fontSize="0.85rem">
              No scanner activity yet. Scanner runs automatically every 15 minutes during market hours (09:15–15:30 IST).
            </Typography>
          </Box>
        ) : (
          <Box sx={{ fontFamily: 'monospace', fontSize: '0.78rem' }}>
            {activity.events.map((e: any, i: number) => (
              <Box key={i} display="flex" alignItems="flex-start" gap={1.5} px={2} py={0.6}
                sx={{ borderBottom: '1px solid rgba(255,255,255,0.04)',
                      '&:hover': { bgcolor: 'rgba(255,255,255,0.03)' } }}>
                <Typography sx={{ color: '#546e7a', minWidth: 60, fontSize: '0.75rem', pt: '1px' }}>
                  {e.ts}
                </Typography>
                <Chip size="small" label={e.event}
                  sx={{ fontSize: '0.65rem', height: 18, minWidth: 72,
                        bgcolor: EVENT_COLORS[e.event] || '#546e7a', color: '#fff' }} />
                {e.symbol && (
                  <Typography sx={{ fontWeight: 700, minWidth: 90, color: '#e0e0e0' }}>
                    {e.symbol}
                  </Typography>
                )}
                {e.verdict && (
                  <Chip size="small" label={e.verdict}
                    sx={{ fontSize: '0.65rem', height: 18,
                          color: e.verdict.includes('BUY') ? '#66bb6a' : e.verdict.includes('SELL') ? '#ef5350' : '#ffd54f' }} />
                )}
                <Typography sx={{ color: '#90a4ae', flex: 1 }}>{e.detail}</Typography>
              </Box>
            ))}
          </Box>
        )}
      </Card>

      <Divider sx={{ mb: 3 }} />

      {/* ── Closed Trades ── */}
      <Typography variant="h6" fontWeight={600} gutterBottom>
        Trade History ({trades.closed?.length || 0})
      </Typography>
      <Box sx={{ overflowX: 'auto' }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Symbol</TableCell>
              <TableCell>Direction</TableCell>
              <TableCell>Entry</TableCell>
              <TableCell>Exit</TableCell>
              <TableCell>Qty</TableCell>
              <TableCell align="right">P&L (₹)</TableCell>
              <TableCell align="right">P&L %</TableCell>
              <TableCell>Reason</TableCell>
              <TableCell>Date</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {[...(trades.closed || [])].reverse().slice(0, 50).map((t: any, i: number) => (
              <TableRow key={i}>
                <TableCell sx={{ fontWeight: 'bold' }}>{t.symbol}</TableCell>
                <TableCell>
                  <Chip size="small" label={t.direction}
                    color={t.direction === 'LONG' ? 'success' : 'error'} />
                </TableCell>
                <TableCell>₹{t.entry_price?.toFixed(2)}</TableCell>
                <TableCell>₹{t.exit_price?.toFixed(2)}</TableCell>
                <TableCell>{t.qty}</TableCell>
                <TableCell align="right"
                  sx={{ color: t.realized_pnl >= 0 ? '#69f0ae' : '#ff5252', fontWeight: 600 }}>
                  {t.realized_pnl >= 0 ? '+' : ''}₹{t.realized_pnl?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                </TableCell>
                <TableCell align="right"
                  sx={{ color: t.realized_pct >= 0 ? '#69f0ae' : '#ff5252' }}>
                  {t.realized_pct >= 0 ? '+' : ''}{t.realized_pct?.toFixed(2)}%
                </TableCell>
                <TableCell>
                  <Chip size="small" label={t.exit_reason || '—'}
                    color={t.exit_reason === 'TP_HIT' ? 'success' : t.exit_reason === 'SL_HIT' ? 'error' : 'default'} />
                </TableCell>
                <TableCell sx={{ fontSize: '0.75rem' }}>
                  {t.exit_time ? new Date(t.exit_time).toLocaleDateString('en-IN') : '—'}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Box>
    </Box>
  );
};

export default StocksPortfolio;
