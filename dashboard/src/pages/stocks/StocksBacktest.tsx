import React, { useState, useEffect } from 'react';
import {
  Box, Grid, Card, CardContent, Typography, Chip,
  CircularProgress, Alert, Table, TableBody, TableCell,
  TableHead, TableRow, Select, MenuItem, FormControl, InputLabel,
} from '@mui/material';

const STOCKS_API = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const VERDICT_COLORS: Record<string, string> = {
  VIABLE: '#69f0ae', MARGINAL: '#ffd54f', NOT_VIABLE: '#ff5252',
};

const StocksBacktest: React.FC = () => {
  const [allResults, setAll] = useState<Record<string, any>>({});
  const [selected, setSelected] = useState<string>('ALL');
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState('');

  useEffect(() => {
    fetch(`${STOCKS_API}/stocks/backtest`)
      .then(r => r.json())
      .then(d => { setAll(d); setLoading(false); })
      .catch(() => { setError('Cannot connect to stocks API'); setLoading(false); });
  }, []);

  if (loading) return (
    <Box display="flex" justifyContent="center" alignItems="center" height="60vh">
      <CircularProgress />
    </Box>
  );
  if (error) return <Box p={3}><Alert severity="warning">{error}</Alert></Box>;

  const symbols  = Object.keys(allResults);
  const detail   = selected !== 'ALL' ? allResults[selected] : null;

  const viable   = symbols.filter(s => allResults[s]?.verdict === 'VIABLE');
  const marginal = symbols.filter(s => allResults[s]?.verdict === 'MARGINAL');
  const notViable = symbols.filter(s => allResults[s]?.verdict === 'NOT_VIABLE');

  return (
    <Box p={3} maxWidth={1400}>
      <Typography variant="h4" fontWeight={700} gutterBottom>Walk-Forward Backtest</Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Expanding WF · LightGBM + Isotonic · Bidirectional · Meta-labeling
      </Typography>

      {symbols.length === 0 ? (
        <Alert severity="info" sx={{ mt: 2 }}>
          No backtest results yet. Run: <code>python walk_forward.py</code> in india-stocks/
        </Alert>
      ) : (
        <>
          {/* ── Summary cards ── */}
          <Grid container spacing={2} sx={{ my: 2 }}>
            {[
              { label: 'VIABLE', syms: viable,    color: '#69f0ae' },
              { label: 'MARGINAL', syms: marginal, color: '#ffd54f' },
              { label: 'NOT VIABLE', syms: notViable, color: '#ff5252' },
            ].map(({ label, syms, color }) => (
              <Grid item xs={12} sm={4} key={label}>
                <Card sx={{ bgcolor: color + '11', border: `1px solid ${color}33`, borderRadius: 2 }}>
                  <CardContent>
                    <Typography variant="overline" sx={{ color }}>{label}</Typography>
                    <Typography variant="h4" fontWeight={700} sx={{ color }}>
                      {syms.length}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {syms.join(', ') || 'None'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          {/* ── All symbols table ── */}
          <Typography variant="h6" fontWeight={600} gutterBottom>All Results</Typography>
          <Box sx={{ overflowX: 'auto', mb: 3 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Verdict</TableCell>
                  <TableCell align="right">Avg Sharpe</TableCell>
                  <TableCell align="right">Avg WR</TableCell>
                  <TableCell align="right">Trades</TableCell>
                  <TableCell align="right">Threshold</TableCell>
                  <TableCell align="right">Folds</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {symbols
                  .sort((a, b) => (allResults[b]?.avg_sharpe || -99) - (allResults[a]?.avg_sharpe || -99))
                  .map(sym => {
                    const r = allResults[sym];
                    const vc = VERDICT_COLORS[r.verdict] || '#616161';
                    return (
                      <TableRow key={sym}
                        onClick={() => setSelected(sym)}
                        sx={{ cursor: 'pointer',
                              bgcolor: selected === sym ? 'rgba(255,255,255,0.05)' : undefined,
                              '&:hover': { bgcolor: 'rgba(255,255,255,0.04)' } }}>
                        <TableCell sx={{ fontWeight: 'bold' }}>{sym}</TableCell>
                        <TableCell>
                          <Chip size="small" label={r.verdict}
                            sx={{ bgcolor: vc + '22', color: vc, fontWeight: 600 }} />
                        </TableCell>
                        <TableCell align="right"
                          sx={{ color: r.avg_sharpe > 0 ? '#69f0ae' : '#ff5252', fontWeight: 600 }}>
                          {r.avg_sharpe?.toFixed(3)}
                        </TableCell>
                        <TableCell align="right">
                          {((r.avg_wr || 0) * 100).toFixed(1)}%
                        </TableCell>
                        <TableCell align="right">{r.total_trades}</TableCell>
                        <TableCell align="right">{r.best_threshold}</TableCell>
                        <TableCell align="right">{r.n_folds}</TableCell>
                      </TableRow>
                    );
                  })}
              </TableBody>
            </Table>
          </Box>

          {/* ── Fold detail for selected symbol ── */}
          {detail && (
            <>
              <Typography variant="h6" fontWeight={600} gutterBottom>
                {selected} — Fold Detail
              </Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Fold</TableCell>
                    <TableCell align="right">Sharpe</TableCell>
                    <TableCell align="right">Win Rate</TableCell>
                    <TableCell align="right">Trades</TableCell>
                    <TableCell align="right">Long</TableCell>
                    <TableCell align="right">Short</TableCell>
                    <TableCell align="right">Avg P&L</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {(detail.fold_results || []).map((f: any) => (
                    <TableRow key={f.fold}>
                      <TableCell>Fold {f.fold}</TableCell>
                      <TableCell align="right"
                        sx={{ color: f.sharpe > 0 ? '#69f0ae' : '#ff5252', fontWeight: 600 }}>
                        {f.sharpe?.toFixed(3)}
                      </TableCell>
                      <TableCell align="right">{((f.wr || 0) * 100).toFixed(1)}%</TableCell>
                      <TableCell align="right">{f.n_trades}</TableCell>
                      <TableCell align="right" sx={{ color: '#69f0ae' }}>{f.long_trades}</TableCell>
                      <TableCell align="right" sx={{ color: '#ff5252' }}>{f.short_trades}</TableCell>
                      <TableCell align="right"
                        sx={{ color: (f.avg_pnl || 0) >= 0 ? '#69f0ae' : '#ff5252' }}>
                        {((f.avg_pnl || 0) * 100).toFixed(2)}%
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </>
          )}
        </>
      )}
    </Box>
  );
};

export default StocksBacktest;
