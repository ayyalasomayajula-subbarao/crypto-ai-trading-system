-- ============================================================
-- Crypto AI Trading System - Portfolio Migration
-- ============================================================
-- Run this SQL in your Supabase SQL Editor AFTER the initial schema
-- https://app.supabase.com/project/_/sql
-- ============================================================

-- ============================================================
-- PORTFOLIO_HOLDINGS TABLE
-- Stores user's coin holdings with real invested amounts
-- ============================================================
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    coin TEXT NOT NULL,
    amount DECIMAL(30, 18) NOT NULL DEFAULT 0,
    avg_price DECIMAL(30, 18) NOT NULL DEFAULT 0,
    total_invested DECIMAL(15, 2) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, coin)
);

-- ============================================================
-- PORTFOLIO_SNAPSHOTS TABLE
-- Daily snapshots for tracking profit/loss over time periods (1D, 7D, 30D)
-- ============================================================
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    total_value DECIMAL(15, 2) NOT NULL,
    total_invested DECIMAL(15, 2) NOT NULL,
    capital DECIMAL(15, 2) NOT NULL,
    holdings_snapshot JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, snapshot_date)
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_user_id ON portfolio_holdings(user_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_user_id ON portfolio_snapshots(user_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date ON portfolio_snapshots(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_user_date ON portfolio_snapshots(user_id, snapshot_date DESC);

-- ============================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================
ALTER TABLE portfolio_holdings ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_snapshots ENABLE ROW LEVEL SECURITY;

-- Portfolio Holdings: Users can only access their own holdings
CREATE POLICY "Users can view own holdings" ON portfolio_holdings
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own holdings" ON portfolio_holdings
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own holdings" ON portfolio_holdings
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own holdings" ON portfolio_holdings
    FOR DELETE USING (auth.uid() = user_id);

-- Portfolio Snapshots: Users can only access their own snapshots
CREATE POLICY "Users can view own snapshots" ON portfolio_snapshots
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own snapshots" ON portfolio_snapshots
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own snapshots" ON portfolio_snapshots
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own snapshots" ON portfolio_snapshots
    FOR DELETE USING (auth.uid() = user_id);

-- ============================================================
-- TRIGGERS
-- ============================================================
CREATE TRIGGER update_portfolio_holdings_updated_at
    BEFORE UPDATE ON portfolio_holdings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Function to upsert a portfolio holding (buy/add coins)
CREATE OR REPLACE FUNCTION upsert_holding(
    p_user_id UUID,
    p_coin TEXT,
    p_amount DECIMAL,
    p_price DECIMAL
)
RETURNS portfolio_holdings AS $$
DECLARE
    v_existing portfolio_holdings;
    v_new_amount DECIMAL;
    v_new_avg_price DECIMAL;
    v_new_total_invested DECIMAL;
    v_result portfolio_holdings;
BEGIN
    -- Get existing holding
    SELECT * INTO v_existing
    FROM portfolio_holdings
    WHERE user_id = p_user_id AND coin = p_coin;

    IF v_existing IS NULL THEN
        -- Insert new holding
        INSERT INTO portfolio_holdings (user_id, coin, amount, avg_price, total_invested)
        VALUES (p_user_id, p_coin, p_amount, p_price, p_amount * p_price)
        RETURNING * INTO v_result;
    ELSE
        -- Update existing: calculate new weighted average price
        v_new_amount := v_existing.amount + p_amount;
        v_new_total_invested := v_existing.total_invested + (p_amount * p_price);
        v_new_avg_price := CASE
            WHEN v_new_amount > 0 THEN v_new_total_invested / v_new_amount
            ELSE 0
        END;

        UPDATE portfolio_holdings
        SET amount = v_new_amount,
            avg_price = v_new_avg_price,
            total_invested = v_new_total_invested,
            updated_at = NOW()
        WHERE id = v_existing.id
        RETURNING * INTO v_result;
    END IF;

    RETURN v_result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to reduce/sell holding
CREATE OR REPLACE FUNCTION reduce_holding(
    p_user_id UUID,
    p_coin TEXT,
    p_amount DECIMAL
)
RETURNS portfolio_holdings AS $$
DECLARE
    v_existing portfolio_holdings;
    v_new_amount DECIMAL;
    v_new_total_invested DECIMAL;
    v_result portfolio_holdings;
BEGIN
    SELECT * INTO v_existing
    FROM portfolio_holdings
    WHERE user_id = p_user_id AND coin = p_coin;

    IF v_existing IS NULL THEN
        RAISE EXCEPTION 'No holding found for coin %', p_coin;
    END IF;

    v_new_amount := GREATEST(0, v_existing.amount - p_amount);

    -- Proportionally reduce total_invested
    v_new_total_invested := CASE
        WHEN v_existing.amount > 0 THEN
            v_existing.total_invested * (v_new_amount / v_existing.amount)
        ELSE 0
    END;

    IF v_new_amount = 0 THEN
        -- Delete the holding if sold completely
        DELETE FROM portfolio_holdings WHERE id = v_existing.id;
        v_result := v_existing;
        v_result.amount := 0;
        v_result.total_invested := 0;
    ELSE
        UPDATE portfolio_holdings
        SET amount = v_new_amount,
            total_invested = v_new_total_invested,
            updated_at = NOW()
        WHERE id = v_existing.id
        RETURNING * INTO v_result;
    END IF;

    RETURN v_result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to record a daily snapshot
CREATE OR REPLACE FUNCTION record_portfolio_snapshot(
    p_user_id UUID,
    p_total_value DECIMAL,
    p_holdings_snapshot JSONB
)
RETURNS portfolio_snapshots AS $$
DECLARE
    v_profile profiles;
    v_total_invested DECIMAL;
    v_result portfolio_snapshots;
BEGIN
    -- Get user's capital and calculate total invested
    SELECT * INTO v_profile FROM profiles WHERE id = p_user_id;

    SELECT COALESCE(SUM(total_invested), 0) INTO v_total_invested
    FROM portfolio_holdings
    WHERE user_id = p_user_id;

    -- Upsert snapshot for today
    INSERT INTO portfolio_snapshots (
        user_id,
        snapshot_date,
        total_value,
        total_invested,
        capital,
        holdings_snapshot
    )
    VALUES (
        p_user_id,
        CURRENT_DATE,
        p_total_value,
        v_total_invested,
        v_profile.capital,
        p_holdings_snapshot
    )
    ON CONFLICT (user_id, snapshot_date)
    DO UPDATE SET
        total_value = EXCLUDED.total_value,
        total_invested = EXCLUDED.total_invested,
        capital = EXCLUDED.capital,
        holdings_snapshot = EXCLUDED.holdings_snapshot
    RETURNING * INTO v_result;

    RETURN v_result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get profit/loss for a time period
CREATE OR REPLACE FUNCTION get_portfolio_pnl(
    p_user_id UUID,
    p_days INTEGER
)
RETURNS TABLE (
    start_value DECIMAL,
    end_value DECIMAL,
    pnl_amount DECIMAL,
    pnl_percent DECIMAL,
    period_start DATE,
    period_end DATE
) AS $$
DECLARE
    v_start_date DATE;
    v_end_snapshot portfolio_snapshots;
    v_start_snapshot portfolio_snapshots;
BEGIN
    v_start_date := CURRENT_DATE - p_days;

    -- Get the latest snapshot (today or most recent)
    SELECT * INTO v_end_snapshot
    FROM portfolio_snapshots
    WHERE user_id = p_user_id
    ORDER BY snapshot_date DESC
    LIMIT 1;

    -- Get the snapshot closest to start date
    SELECT * INTO v_start_snapshot
    FROM portfolio_snapshots
    WHERE user_id = p_user_id
      AND snapshot_date <= v_start_date
    ORDER BY snapshot_date DESC
    LIMIT 1;

    -- If no start snapshot, use the oldest available
    IF v_start_snapshot IS NULL THEN
        SELECT * INTO v_start_snapshot
        FROM portfolio_snapshots
        WHERE user_id = p_user_id
        ORDER BY snapshot_date ASC
        LIMIT 1;
    END IF;

    -- Return results
    IF v_end_snapshot IS NOT NULL AND v_start_snapshot IS NOT NULL THEN
        start_value := v_start_snapshot.total_value;
        end_value := v_end_snapshot.total_value;
        pnl_amount := v_end_snapshot.total_value - v_start_snapshot.total_value;
        pnl_percent := CASE
            WHEN v_start_snapshot.total_value > 0 THEN
                ((v_end_snapshot.total_value - v_start_snapshot.total_value) / v_start_snapshot.total_value) * 100
            ELSE 0
        END;
        period_start := v_start_snapshot.snapshot_date;
        period_end := v_end_snapshot.snapshot_date;
        RETURN NEXT;
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================
-- DONE!
-- ============================================================
-- Run this migration after the initial schema.
-- Your portfolio tracking is now set up!
-- ============================================================
