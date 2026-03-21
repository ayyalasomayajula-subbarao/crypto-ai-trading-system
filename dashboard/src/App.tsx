import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import theme from './theme';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Layout from './components/layout/Layout';
import Login from './components/Login';
import Dashboard from './components/Dashboard';
import CoinPage from './components/Coinpage';
import Backtest from './components/Backtest';
import SignalHistory from './components/SignalHistory';
import PaperTrading from './components/PaperTrading';
import CoinsPage from './pages/CoinsPage';
import PortfolioPage from './pages/PortfolioPage';
import SettingsPage from './pages/SettingsPage';

// India Stocks pages
import StocksDashboard from './pages/stocks/StocksDashboard';
import StocksPage from './pages/stocks/StocksPage';
import StockDetail from './pages/stocks/StockDetail';
import MarketPulse from './pages/stocks/MarketPulse';
import StocksPortfolio from './pages/stocks/StocksPortfolio';
import StocksBacktest from './pages/stocks/StocksBacktest';
import BrokerConnect from './pages/stocks/BrokerConnect';

import './App.css';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AuthProvider>
          <Routes>
            {/* Public routes */}
            <Route path="/login" element={<Login />} />

            {/* Protected routes with sidebar layout */}
            <Route
              element={
                <ProtectedRoute>
                  <Layout />
                </ProtectedRoute>
              }
            >
              {/* ── Crypto routes (existing) ── */}
              <Route path="/" element={<Dashboard />} />
              <Route path="/portfolio" element={<PortfolioPage />} />
              <Route path="/coins" element={<CoinsPage />} />
              <Route path="/coin/:coinId" element={<CoinPage />} />
              <Route path="/backtest" element={<Backtest />} />
              <Route path="/signals" element={<SignalHistory />} />
              <Route path="/paper-trading" element={<PaperTrading />} />
              <Route path="/settings" element={<SettingsPage />} />

              {/* ── India Stocks routes (new) ── */}
              <Route path="/stocks" element={<StocksDashboard />} />
              <Route path="/stocks/screener" element={<StocksPage />} />
              <Route path="/stocks/:symbol" element={<StockDetail />} />
              <Route path="/stocks/market/pulse" element={<MarketPulse />} />
              <Route path="/stocks/portfolio" element={<StocksPortfolio />} />
              <Route path="/stocks/backtest" element={<StocksBacktest />} />
              <Route path="/stocks/broker" element={<BrokerConnect />} />
            </Route>
          </Routes>
        </AuthProvider>
      </Router>
    </ThemeProvider>
  );
}

export default App;
