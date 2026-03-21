import React, { useState, useEffect } from 'react';
import {
  Box, Grid, Card, CardContent, Typography, Chip,
  CircularProgress, Alert, Divider, LinearProgress,
  Table, TableBody, TableCell, TableHead, TableRow,
} from '@mui/material';
import './MarketPulse.css';

const STOCKS_API = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const VIX_COLORS: Record<string, string> = {
  EXTREME_FEAR: '#ff1744', HIGH_FEAR: '#ff5722',
  NEUTRAL: '#ffd54f', COMPLACENCY: '#69f0ae',
};

const MarketPulse: React.FC = () => {
  const [overview, setOverview] = useState<any>(null);
  const [fiiDii, setFiiDii]     = useState<any[]>([]);
  const [vix, setVix]           = useState<any[]>([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState('');

  useEffect(() => {
    const fetch_ = async () => {
      try {
        const [ovRes, fRes, vRes] = await Promise.all([
          fetch(`${STOCKS_API}/stocks/market/overview`),
          fetch(`${STOCKS_API}/stocks/fii-dii?days=30`),
          fetch(`${STOCKS_API}/stocks/india-vix?days=60`),
        ]);
        setOverview(await ovRes.json());
        const fData = await fRes.json();
        setFiiDii(fData.data || []);
        const vData = await vRes.json();
        setVix(vData.data || []);
      } catch (e: any) {
        setError('Cannot connect to stocks API (port 8000)');
      } finally {
        setLoading(false);
      }
    };
    fetch_();
    const id = setInterval(fetch_, 120_000);
    return () => clearInterval(id);
  }, []);

  if (loading) return (
    <Box display="flex" justifyContent="center" alignItems="center" height="60vh">
      <CircularProgress />
    </Box>
  );
  if (error) return <Box p={3}><Alert severity="warning">{error}</Alert></Box>;

  const oc = overview?.option_chain || {};
  const vd = overview?.india_vix;
  const fd = overview?.fii_dii;

  return (
    <Box p={3} maxWidth={1400}>
      <Typography variant="h4" fontWeight={700} gutterBottom>Market Pulse</Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Real-time Indian market internals — FII/DII · VIX · PCR · Options
      </Typography>

      <Grid container spacing={3} sx={{ mt: 1 }}>
        {/* ── India VIX ── */}
        <Grid item xs={12} md={4}>
          <Card className="pulse-section-card">
            <CardContent>
              <Typography variant="overline" color="text.secondary">INDIA VIX</Typography>
              {vd ? (
                <>
                  <Box display="flex" alignItems="baseline" gap={1.5} mt={1}>
                    <Typography variant="h2" fontWeight={800}
                      sx={{ color: VIX_COLORS[vd.level] || '#ffd54f' }}>
                      {vd.value.toFixed(1)}
                    </Typography>
                    <Typography variant="body1"
                      sx={{ color: vd.change_pct >= 0 ? '#ff5252' : '#69f0ae' }}>
                      {vd.change_pct >= 0 ? '+' : ''}{vd.change_pct.toFixed(2)}%
                    </Typography>
                  </Box>
                  <Chip label={vd.level.replace(/_/g,' ')}
                    sx={{ bgcolor: (VIX_COLORS[vd.level] || '#ffd54f') + '22',
                          color: VIX_COLORS[vd.level] || '#ffd54f', mt: 1 }} />
                  <Divider sx={{ my: 2 }} />
                  {/* VIX mini chart */}
                  <Typography variant="caption" color="text.secondary">60-day VIX</Typography>
                  <MiniLineChart data={vix.map(d => d.vix)} color="#ffd54f" />
                </>
              ) : (
                <Typography color="text.secondary">No VIX data</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* ── FII/DII ── */}
        <Grid item xs={12} md={4}>
          <Card className="pulse-section-card">
            <CardContent>
              <Typography variant="overline" color="text.secondary">FII / DII FLOWS (₹ Cr)</Typography>
              {fd ? (
                <>
                  <Grid container spacing={1} mt={1}>
                    {[
                      { label: 'FII Today', val: fd.fii_net, col: fd.fii_net >= 0 ? '#69f0ae' : '#ff5252' },
                      { label: 'DII Today', val: fd.dii_net, col: fd.dii_net >= 0 ? '#69f0ae' : '#ff5252' },
                      { label: 'FII 7d',   val: fd.fii_7d_cumulative, col: fd.fii_7d_cumulative >= 0 ? '#69f0ae' : '#ff5252' },
                      { label: 'DII 7d',   val: fd.dii_7d_cumulative, col: fd.dii_7d_cumulative >= 0 ? '#69f0ae' : '#ff5252' },
                    ].map(({ label, val, col }) => (
                      <Grid item xs={6} key={label}>
                        <Box sx={{ bgcolor: col + '11', borderRadius: 2, p: 1.5, border: `1px solid ${col}22` }}>
                          <Typography variant="caption" color="text.secondary">{label}</Typography>
                          <Typography variant="body2" fontWeight={700} sx={{ color: col }}>
                            {val >= 0 ? '+' : ''}{val?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                          </Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="caption" color="text.secondary">30-day FII Net Flow</Typography>
                  <DualBarChart data={fiiDii} />
                </>
              ) : (
                <Typography color="text.secondary">No FII/DII data</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* ── Option Chain Metrics ── */}
        <Grid item xs={12} md={4}>
          <Card className="pulse-section-card">
            <CardContent>
              <Typography variant="overline" color="text.secondary">OPTION CHAIN</Typography>
              {Object.entries(oc).map(([sym, data]: any) => (
                <Box key={sym} mb={2}>
                  <Typography variant="body2" fontWeight={600} color="primary.main" gutterBottom>
                    {sym}
                  </Typography>
                  <Grid container spacing={1}>
                    {[
                      { label: 'PCR', val: data?.pcr?.toFixed(2) },
                      { label: 'PCR 7d avg', val: data?.pcr_7d_avg?.toFixed(2) },
                      { label: 'Max Pain', val: `₹${data?.max_pain?.toFixed(0)}` },
                      { label: 'IV %ile', val: `${data?.iv_percentile?.toFixed(0)}%` },
                    ].map(({ label, val }) => (
                      <Grid item xs={6} key={label}>
                        <Box sx={{ bgcolor: 'rgba(255,255,255,0.04)', borderRadius: 1, p: 1 }}>
                          <Typography variant="caption" color="text.secondary" display="block">{label}</Typography>
                          <Typography variant="body2" fontWeight={600}>{val || '—'}</Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              ))}
              {Object.keys(oc).length === 0 && (
                <Typography color="text.secondary" variant="body2">
                  Run collect_option_chain.py to populate
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* ── FII/DII 30-day table ── */}
        <Grid item xs={12}>
          <Card className="pulse-section-card">
            <CardContent>
              <Typography variant="overline" color="text.secondary">30-Day FII/DII Activity</Typography>
              <Box sx={{ overflowX: 'auto', mt: 1 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Date</TableCell>
                      <TableCell align="right">FII Net (₹ Cr)</TableCell>
                      <TableCell align="right">DII Net (₹ Cr)</TableCell>
                      <TableCell align="right">FII 7d Cumul</TableCell>
                      <TableCell align="right">Divergence</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {[...fiiDii].reverse().slice(0, 20).map((row: any, i: number) => (
                      <TableRow key={i}>
                        <TableCell sx={{ fontSize: '0.75rem' }}>{row.date}</TableCell>
                        <TableCell align="right"
                          sx={{ color: row.fii_net >= 0 ? '#69f0ae' : '#ff5252',
                                fontWeight: 500, fontSize: '0.8rem' }}>
                          {row.fii_net >= 0 ? '+' : ''}{row.fii_net?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                        </TableCell>
                        <TableCell align="right"
                          sx={{ color: row.dii_net >= 0 ? '#69f0ae' : '#ff5252',
                                fontWeight: 500, fontSize: '0.8rem' }}>
                          {row.dii_net >= 0 ? '+' : ''}{row.dii_net?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                        </TableCell>
                        <TableCell align="right" sx={{ fontSize: '0.8rem' }}>
                          {row.fii_7d_cumul?.toLocaleString('en-IN', { maximumFractionDigits: 0 }) || '—'}
                        </TableCell>
                        <TableCell align="right" sx={{ fontSize: '0.8rem' }}>
                          {row.divergence?.toLocaleString('en-IN', { maximumFractionDigits: 0 }) || '—'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

/* ── Mini charts ──────────────────────────────────────────────────────────── */
const MiniLineChart: React.FC<{ data: number[]; color: string }> = ({ data, color }) => {
  if (!data.length) return null;
  const W = 300; const H = 60; const PAD = 4;
  const min = Math.min(...data); const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = PAD + (i / (data.length - 1)) * (W - PAD * 2);
    const y = PAD + (1 - (v - min) / range) * (H - PAD * 2);
    return `${x},${y}`;
  }).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', marginTop: 4 }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
};

const DualBarChart: React.FC<{ data: any[] }> = ({ data }) => {
  if (!data.length) return null;
  const last20 = data.slice(-20);
  const maxAbs = Math.max(...last20.map(d => Math.abs(d.fii_net || 0)));
  if (maxAbs === 0) return null;
  const W = 300; const H = 80;
  const barW = (W / last20.length) * 0.8;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', marginTop: 4 }}>
      <line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
      {last20.map((d, i) => {
        const x  = (i / last20.length) * W + barW * 0.1;
        const v  = d.fii_net || 0;
        const h  = Math.abs(v) / maxAbs * (H / 2 - 4);
        const y  = v >= 0 ? H / 2 - h : H / 2;
        const cl = v >= 0 ? '#69f0ae' : '#ff5252';
        return <rect key={i} x={x} y={y} width={barW} height={h}
          fill={cl} opacity={0.7} rx={1} />;
      })}
    </svg>
  );
};

export default MarketPulse;
