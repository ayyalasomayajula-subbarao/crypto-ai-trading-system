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
              <Route path="/" element={<Dashboard />} />
              <Route path="/portfolio" element={<PortfolioPage />} />
              <Route path="/coins" element={<CoinsPage />} />
              <Route path="/coin/:coinId" element={<CoinPage />} />
              <Route path="/backtest" element={<Backtest />} />
              <Route path="/signals" element={<SignalHistory />} />
              <Route path="/paper-trading" element={<PaperTrading />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Route>
          </Routes>
        </AuthProvider>
      </Router>
    </ThemeProvider>
  );
}

export default App;
