-- ============================================================
-- Crypto AI Trading System - Supabase Schema
-- ============================================================
-- Run this SQL in your Supabase SQL Editor:
-- https://app.supabase.com/project/_/sql
-- ============================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- PROFILES TABLE
-- Stores user profile information
-- ============================================================
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT UNIQUE NOT NULL,
    username TEXT UNIQUE NOT NULL,
    display_name TEXT,
    avatar_url TEXT,
    capital DECIMAL(15, 2) DEFAULT 1000.00,
    experience_level TEXT DEFAULT 'INTERMEDIATE' CHECK (experience_level IN ('BEGINNER', 'INTERMEDIATE', 'ADVANCED')),
    default_trade_type TEXT DEFAULT 'SWING' CHECK (default_trade_type IN ('SCALP', 'SHORT_TERM', 'SWING', 'INVESTMENT')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- TRADES TABLE
-- Stores user trade history
-- ============================================================
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    coin TEXT NOT NULL,
    trade_type TEXT NOT NULL CHECK (trade_type IN ('SCALP', 'SHORT_TERM', 'SWING', 'INVESTMENT')),
    entry_price DECIMAL(20, 10) NOT NULL,
    exit_price DECIMAL(20, 10),
    amount DECIMAL(20, 10) NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
    status TEXT DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED')),
    pnl DECIMAL(15, 2),
    pnl_percent DECIMAL(10, 4),
    stop_loss DECIMAL(20, 10),
    take_profit DECIMAL(20, 10),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

-- ============================================================
-- WATCHLIST TABLE
-- Stores user watchlist items
-- ============================================================
CREATE TABLE IF NOT EXISTS watchlist (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    coin TEXT NOT NULL,
    alerts_enabled BOOLEAN DEFAULT FALSE,
    target_price DECIMAL(20, 10),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, coin)
);

-- ============================================================
-- TRADE_SIGNALS TABLE
-- Stores AI-generated signals history
-- ============================================================
CREATE TABLE IF NOT EXISTS trade_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
    coin TEXT NOT NULL,
    trade_type TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confidence TEXT,
    win_probability DECIMAL(5, 2),
    loss_probability DECIMAL(5, 2),
    expectancy DECIMAL(10, 2),
    price_at_signal DECIMAL(20, 10),
    reasoning JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- USER_SETTINGS TABLE
-- Stores user preferences and settings
-- ============================================================
CREATE TABLE IF NOT EXISTS user_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID UNIQUE NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    theme TEXT DEFAULT 'dark',
    notifications_enabled BOOLEAN DEFAULT TRUE,
    email_alerts BOOLEAN DEFAULT FALSE,
    default_capital DECIMAL(15, 2) DEFAULT 1000.00,
    risk_tolerance TEXT DEFAULT 'MEDIUM' CHECK (risk_tolerance IN ('LOW', 'MEDIUM', 'HIGH')),
    preferred_coins TEXT[] DEFAULT ARRAY['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT'],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_watchlist_user_id ON watchlist(user_id);
CREATE INDEX IF NOT EXISTS idx_trade_signals_created_at ON trade_signals(created_at DESC);

-- ============================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================

-- Enable RLS on all tables
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;

-- Profiles: Users can only read/update their own profile
CREATE POLICY "Users can view own profile" ON profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile" ON profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

-- Trades: Users can only access their own trades
CREATE POLICY "Users can view own trades" ON trades
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own trades" ON trades
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own trades" ON trades
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own trades" ON trades
    FOR DELETE USING (auth.uid() = user_id);

-- Watchlist: Users can only access their own watchlist
CREATE POLICY "Users can view own watchlist" ON watchlist
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own watchlist" ON watchlist
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own watchlist" ON watchlist
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own watchlist" ON watchlist
    FOR DELETE USING (auth.uid() = user_id);

-- Trade Signals: Users can view their own signals, or signals with no user
CREATE POLICY "Users can view own signals" ON trade_signals
    FOR SELECT USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can insert signals" ON trade_signals
    FOR INSERT WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

-- User Settings: Users can only access their own settings
CREATE POLICY "Users can view own settings" ON user_settings
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own settings" ON user_settings
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own settings" ON user_settings
    FOR UPDATE USING (auth.uid() = user_id);

-- ============================================================
-- FUNCTIONS
-- ============================================================

-- Function to handle new user signup
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO profiles (id, email, username, display_name)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'username', split_part(NEW.email, '@', 1)),
        COALESCE(NEW.raw_user_meta_data->>'display_name', NEW.raw_user_meta_data->>'username', split_part(NEW.email, '@', 1))
    );

    INSERT INTO user_settings (user_id)
    VALUES (NEW.id);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to automatically create profile on signup
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_profiles_updated_at
    BEFORE UPDATE ON profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_user_settings_updated_at
    BEFORE UPDATE ON user_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- SAMPLE DATA (Optional - Comment out if not needed)
-- ============================================================

-- You can insert sample data here for testing
-- INSERT INTO ...

-- ============================================================
-- DONE!
-- ============================================================
-- Your Supabase database is now set up for the Crypto AI Trading System.
--
-- Next steps:
-- 1. Copy your Supabase URL and anon key from Settings > API
-- 2. Create a .env file in dashboard/ with:
--    REACT_APP_SUPABASE_URL=your-url
--    REACT_APP_SUPABASE_ANON_KEY=your-key
-- 3. Restart your React app
-- ============================================================
