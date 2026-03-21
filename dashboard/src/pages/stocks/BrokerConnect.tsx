import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, Typography, TextField, Button,
  Alert, Chip, Divider, Grid,
} from '@mui/material';
import LockIcon from '@mui/icons-material/Lock';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const STOCKS_API = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const BrokerConnect: React.FC = () => {
  const [connected, setConnected] = useState(false);
  const [funds, setFunds]         = useState<any>(null);
  const [form, setForm]           = useState({
    api_key: '', client_id: '', password: '', totp_secret: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    fetch(`${STOCKS_API}/stocks/broker/status`)
      .then(r => r.json())
      .then(d => { setConnected(d.connected); setFunds(d.funds); })
      .catch(() => {});
  }, []);

  const handleConnect = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true); setError(''); setSuccess('');
    try {
      const res = await fetch(`${STOCKS_API}/stocks/broker/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (data.ok) {
        setSuccess('Connected to Angel One successfully!');
        setConnected(true);
      } else {
        setError(data.detail || 'Connection failed');
      }
    } catch {
      setError('Cannot connect to stocks API');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box p={3} maxWidth={800}>
      <Typography variant="h4" fontWeight={700} gutterBottom>Broker Connect</Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Connect Angel One for live trading (FREE API). Paper trading works without a broker.
      </Typography>

      {/* Status */}
      <Card sx={{ bgcolor: connected ? 'rgba(0,230,118,0.08)' : 'rgba(255,255,255,0.03)',
                  border: `1px solid ${connected ? '#00e676' : 'rgba(255,255,255,0.08)'}`,
                  borderRadius: 2, mb: 3, mt: 2 }}>
        <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {connected
            ? <CheckCircleIcon sx={{ color: '#00e676', fontSize: 32 }} />
            : <LockIcon sx={{ color: '#9e9e9e', fontSize: 32 }} />}
          <Box flex={1}>
            <Typography fontWeight={600}>
              {connected ? 'Angel One Connected' : 'No Broker Connected'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {connected ? 'Live trading enabled' : 'Paper trading mode active'}
            </Typography>
          </Box>
          {funds && (
            <Box textAlign="right">
              <Typography variant="caption" color="text.secondary">Available Funds</Typography>
              <Typography variant="h6" fontWeight={700} color="#69f0ae">
                ₹{funds.available?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Setup instructions */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2" fontWeight={600} gutterBottom>Setup Steps:</Typography>
        <ol style={{ margin: 0, paddingLeft: 20 }}>
          <li>Open an Angel One demat account (free)</li>
          <li>Visit smartapi.angelbroking.com → Create API key</li>
          <li>Enable TOTP in your Angel One app for 2FA</li>
          <li>Enter credentials below (stored in session only, not saved)</li>
        </ol>
      </Alert>

      {/* Connect form */}
      <Card sx={{ bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 2 }}>
        <CardContent>
          <Typography variant="h6" fontWeight={600} gutterBottom>Angel One Credentials</Typography>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}
          <form onSubmit={handleConnect}>
            <Grid container spacing={2}>
              {[
                { key: 'api_key',     label: 'API Key',     help: 'From smartapi.angelbroking.com' },
                { key: 'client_id',  label: 'Client ID',   help: 'Your Angel One client code' },
                { key: 'password',   label: 'Password',    help: 'Your Angel One login password', type: 'password' },
                { key: 'totp_secret', label: 'TOTP Secret', help: 'Base32 secret from authenticator setup', type: 'password' },
              ].map(({ key, label, help, type }) => (
                <Grid item xs={12} sm={6} key={key}>
                  <TextField
                    fullWidth size="small" label={label}
                    type={type || 'text'}
                    value={form[key as keyof typeof form]}
                    onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
                    helperText={help}
                    required
                  />
                </Grid>
              ))}
            </Grid>
            <Button type="submit" variant="contained" fullWidth disabled={loading}
              sx={{ mt: 2, fontWeight: 700, py: 1.2 }}>
              {loading ? 'Connecting...' : 'Connect Angel One'}
            </Button>
          </form>
        </CardContent>
      </Card>

      <Divider sx={{ my: 3 }} />

      {/* Alternative brokers */}
      <Typography variant="h6" fontWeight={600} gutterBottom>Other Brokers (Coming Soon)</Typography>
      <Grid container spacing={2}>
        {[
          { name: 'Zerodha Kite', cost: '₹2000/month', desc: 'Best documentation, most popular' },
          { name: 'Upstox',       cost: 'Free API',     desc: 'Good for algo trading' },
          { name: 'Fyers',        cost: 'Free API',     desc: 'Reliable REST API' },
          { name: 'Dhan HQ',      cost: 'Free API',     desc: 'Modern, fast WebSocket' },
        ].map(b => (
          <Grid item xs={12} sm={6} key={b.name}>
            <Card sx={{ bgcolor: 'rgba(255,255,255,0.02)', borderRadius: 2 }}>
              <CardContent sx={{ py: 1.5 }}>
                <Box display="flex" justifyContent="space-between">
                  <Typography fontWeight={600}>{b.name}</Typography>
                  <Chip size="small" label={b.cost} variant="outlined" />
                </Box>
                <Typography variant="body2" color="text.secondary">{b.desc}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default BrokerConnect;
