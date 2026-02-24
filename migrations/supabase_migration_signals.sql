-- ============================================================
-- Crypto AI Trading System - Signal History Migration
-- ============================================================
-- Run this SQL in your Supabase SQL Editor AFTER the initial schema
-- Adds columns to trade_signals for enhanced signal tracking
-- ============================================================

-- Add new columns to trade_signals for signal history tracking
ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS trade_type TEXT;
ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS sideways_probability DECIMAL(5, 2);
ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS market_regime TEXT;
ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS adx_value DECIMAL(5, 2);

-- Add index on coin for faster filtering
CREATE INDEX IF NOT EXISTS idx_trade_signals_coin ON trade_signals(coin);
CREATE INDEX IF NOT EXISTS idx_trade_signals_verdict ON trade_signals(verdict);
CREATE INDEX IF NOT EXISTS idx_trade_signals_user_coin ON trade_signals(user_id, coin);

-- ============================================================
-- DONE!
-- ============================================================
-- After running this migration:
-- 1. Signal history will be auto-saved from the CoinPage analysis
-- 2. View signal history at /signals in the dashboard
-- 3. Run backtests at /backtest in the dashboard
-- ============================================================
