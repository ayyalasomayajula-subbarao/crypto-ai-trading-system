import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardActionArea,
  CardContent,
  Typography,
  Chip,
  Skeleton,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || window.location.origin;
const WS_URL = process.env.REACT_APP_WS_URL || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/prices`;

interface PriceData {
  price: number;
  change_24h: number;
  direction: 'up' | 'down' | 'same';
}

interface Signal {
  coin: string;
  verdict: string;
  confidence: string;
  win_probability: number | null;
}

const COINS = [
  { key: 'BTC_USDT', name: 'Bitcoin', symbol: 'BTC', icon: '₿', color: '#f7931a' },
  { key: 'ETH_USDT', name: 'Ethereum', symbol: 'ETH', icon: 'Ξ', color: '#627eea' },
  { key: 'SOL_USDT', name: 'Solana', symbol: 'SOL', icon: '◎', color: '#00ffa3' },
  { key: 'PEPE_USDT', name: 'Pepe', symbol: 'PEPE', icon: '🐸', color: '#4a9c2d' },
];

const formatPrice = (price: number): string => {
  if (price < 0.0001) return `$${price.toFixed(8)}`;
  if (price < 0.01) return `$${price.toFixed(6)}`;
  if (price < 1) return `$${price.toFixed(4)}`;
  if (price < 1000) return `$${price.toFixed(2)}`;
  return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

const CoinsPage: React.FC = () => {
  const navigate = useNavigate();
  const [prices, setPrices] = useState<Record<string, PriceData>>({});
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);
  const wsRef = useRef<WebSocket | null>(null);

  // WebSocket for live prices
  useEffect(() => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => setLoading(false);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'initial' || data.type === 'all_prices') {
        setPrices(data.prices);
        setLoading(false);
      } else if (data.type === 'price_update') {
        setPrices(prev => ({
          ...prev,
          [data.coin]: data.data,
        }));
      }
    };

    ws.onclose = () => {
      setTimeout(() => {
        wsRef.current = new WebSocket(WS_URL);
      }, 3000);
    };

    wsRef.current = ws;
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, []);

  // Fetch scan signals
  useEffect(() => {
    const fetchSignals = async () => {
      try {
        const response = await axios.get(`${API_BASE}/scan`);
        setSignals(response.data.signals);
      } catch (err) {
        console.error('Error fetching signals:', err);
      }
    };
    fetchSignals();
  }, []);

  const getSignal = (coinKey: string) => signals.find(s => s.coin === coinKey);

  const verdictColor = (verdict: string) => {
    switch (verdict) {
      case 'BUY': return '#10b981';
      case 'WAIT': return '#f59e0b';
      case 'AVOID': return '#ef4444';
      default: return '#94a3b8';
    }
  };

  return (
    <Box>
      <Typography variant="h5" sx={{ fontWeight: 700, mb: 3, color: 'text.primary' }}>
        Coins
      </Typography>

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', lg: 'repeat(4, 1fr)' },
          gap: 3,
        }}
      >
        {COINS.map((coin) => {
          const priceData = prices[coin.key];
          const signal = getSignal(coin.key);
          const change = priceData?.change_24h ?? 0;
          const isUp = change >= 0;

          return (
            <Card key={coin.key} sx={{ position: 'relative', overflow: 'visible' }}>
              <CardActionArea
                onClick={() => navigate(`/coin/${coin.key.toLowerCase()}`)}
                sx={{ p: 0 }}
              >
                <CardContent sx={{ p: 3 }}>
                  {/* Coin header */}
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                      <Typography
                        sx={{
                          fontSize: '1.75rem',
                          width: 44,
                          height: 44,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          borderRadius: '12px',
                          backgroundColor: `${coin.color}15`,
                          color: coin.color,
                        }}
                      >
                        {coin.icon}
                      </Typography>
                      <Box>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600, color: 'text.primary', lineHeight: 1.2 }}>
                          {coin.symbol}
                        </Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                          {coin.name}
                        </Typography>
                      </Box>
                    </Box>

                    {signal && (
                      <Chip
                        label={signal.verdict}
                        size="small"
                        sx={{
                          backgroundColor: `${verdictColor(signal.verdict)}20`,
                          color: verdictColor(signal.verdict),
                          fontWeight: 700,
                          fontSize: '0.7rem',
                        }}
                      />
                    )}
                  </Box>

                  {/* Price */}
                  {loading || !priceData ? (
                    <Skeleton variant="text" width="60%" height={36} />
                  ) : (
                    <Typography variant="h5" sx={{ fontWeight: 700, color: 'text.primary', mb: 0.5 }}>
                      {formatPrice(priceData.price)}
                    </Typography>
                  )}

                  {/* 24h change */}
                  {priceData && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      {isUp ? (
                        <TrendingUpIcon sx={{ fontSize: 18, color: '#10b981' }} />
                      ) : (
                        <TrendingDownIcon sx={{ fontSize: 18, color: '#ef4444' }} />
                      )}
                      <Typography
                        variant="body2"
                        sx={{ color: isUp ? '#10b981' : '#ef4444', fontWeight: 600 }}
                      >
                        {isUp ? '+' : ''}{change.toFixed(2)}%
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'text.secondary', ml: 0.5 }}>
                        24h
                      </Typography>
                    </Box>
                  )}

                  {/* Win probability */}
                  {signal?.win_probability != null && (
                    <Box sx={{ mt: 1.5, pt: 1.5, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                        Win Probability
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                        <Box
                          sx={{
                            flex: 1,
                            height: 6,
                            borderRadius: 3,
                            backgroundColor: 'rgba(255,255,255,0.06)',
                            overflow: 'hidden',
                          }}
                        >
                          <Box
                            sx={{
                              width: `${signal.win_probability}%`,
                              height: '100%',
                              borderRadius: 3,
                              backgroundColor: signal.win_probability > 60 ? '#10b981' : signal.win_probability > 45 ? '#f59e0b' : '#ef4444',
                            }}
                          />
                        </Box>
                        <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 600, minWidth: 36 }}>
                          {signal.win_probability.toFixed(0)}%
                        </Typography>
                      </Box>
                    </Box>
                  )}
                </CardContent>
              </CardActionArea>
            </Card>
          );
        })}
      </Box>
    </Box>
  );
};

export default CoinsPage;
