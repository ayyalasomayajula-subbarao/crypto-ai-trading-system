# Implementation Plan: Backtesting Engine + Signal History

## Overview
Add two major features:
1. **Backtesting Engine** - Walk-forward validation with Sharpe ratio, max drawdown, equity curve
2. **Signal/Prediction History** - Auto-log every analysis, track model accuracy over time

## Architecture Decision
- **Keep DB frontend-only** (no Supabase on backend) - avoids adding service-role key complexity
- Backtesting runs entirely from historical CSVs on backend, returns JSON results to frontend
- Signal history stored in Supabase from frontend after each analysis call
- This matches the existing pattern where all Supabase writes happen in React

---

## Phase 1: Backend - Backtesting Engine

### New file: `backtesting_engine.py`
Build on existing `backtester_realistic.py` logic, adding:

- **`BacktestingEngine` class** with configurable params (threshold, tp%, sl%, position_size, capital)
- **`run_backtest(coin, threshold)`** - Same core logic as RealisticBacktester but returns:
  - Timestamped equity curve (for charting)
  - Sharpe ratio (annualized from daily returns)
  - Max drawdown
  - Win rate by market regime (using ADX from feature data)
  - Individual trade log with entry/exit timestamps
- **`run_walk_forward(coin, n_splits=5)`** - Expanding window train/test splits
- **`run_all_coins(threshold)`** - Batch all 4 coins
- Downsample equity curve to max ~500 points for API response size

### Modify: `api_final.py`
Add 3 new endpoints:

```
GET /backtest/{coin}?threshold=0.45&capital=10000
GET /backtest?threshold=0.45&capital=10000          (all coins)
GET /backtest/walk-forward/{coin}?splits=5
```

Response shape:
```json
{
  "coin": "BTC_USDT",
  "metrics": {
    "total_return_pct": 42.3,
    "win_rate": 61.5,
    "sharpe_ratio": 1.82,
    "max_drawdown_pct": -8.4,
    "profit_factor": 2.1,
    "total_trades": 47
  },
  "equity_curve": [{"timestamp": "2024-06-01T00:00:00", "equity": 10000}, ...],
  "trades": [...],
  "regime_stats": {"TRENDING": {"trades": 20, "win_rate": 72.0}, ...},
  "config": {"threshold": 0.45, "tp_pct": 5, "sl_pct": 3, ...}
}
```

---

## Phase 2: Frontend - Backtest Page

### New file: `dashboard/src/components/Backtest.tsx`
### New file: `dashboard/src/components/Backtest.css`

**Layout:**
1. **Controls bar** - Coin selector (+ "All"), threshold slider (0.35-0.60), capital input, "Run Backtest" button
2. **Metrics cards row** - Total Return, Win Rate, Sharpe Ratio, Max Drawdown, Profit Factor, Total Trades
3. **Equity curve chart** - Using `lightweight-charts` (already installed), follow PriceChart.tsx pattern
4. **Regime breakdown** - Table showing win rate per market regime
5. **Trade log** - Collapsible table of individual trades with entry/exit, PnL, exit reason

### Modify: `App.tsx`
Add route: `/backtest` → `<Backtest />`

### Modify: `Dashboard.tsx`
Add "Backtest" navigation button in the header area

---

## Phase 3: Frontend - Signal History

### New file: `dashboard/src/components/SignalHistory.tsx`
### New file: `dashboard/src/components/SignalHistory.css`

**Approach:** After each `/analyze/{coin}` call on CoinPage, save the signal to Supabase `trade_signals` table from the frontend. This follows the existing pattern (portfolio writes happen in React).

**Modify: `dashboard/src/components/Coinpage.tsx`**
- After successful analysis, auto-save signal to `trade_signals` via Supabase client
- Fields: coin, verdict, win/loss probability, price_at_signal, reasoning (JSONB), trade_type

**SignalHistory page layout:**
1. **Accuracy summary cards** per coin - total signals, BUY/WAIT/AVOID counts
2. **Signal history table** - Coin, verdict (color-coded badge), probabilities, price, timestamp
3. **Filters** - By coin, by verdict, date range

### Modify: `App.tsx`
Add route: `/signals` → `<SignalHistory />`

### Modify: `Dashboard.tsx`
Add "Signal History" navigation button

---

## Phase 4: Supabase Schema Update

### New file: `migrations/supabase_migration_signals.sql`

Alter `trade_signals` to add tracking columns:
```sql
ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS trade_type TEXT;
ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS sideways_probability DECIMAL(5, 2);
ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS market_regime TEXT;
ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS adx_value DECIMAL(5, 2);
```

No new tables needed - `trade_signals` already has the right structure for signal history.
Backtest results are ephemeral (computed on-demand), no need to persist.

---

## Implementation Order
1. `backtesting_engine.py` (standalone, no deps on other new code)
2. API endpoints in `api_final.py` (wire up backtesting engine)
3. `Backtest.tsx` + `Backtest.css` (frontend backtest page)
4. `App.tsx` + `Dashboard.tsx` routing updates
5. Signal logging in `Coinpage.tsx`
6. `SignalHistory.tsx` + `SignalHistory.css`
7. Migration SQL file

## Files Modified
- `api_final.py` - Add backtest endpoints (~50 lines)
- `dashboard/src/App.tsx` - Add 2 routes
- `dashboard/src/components/Dashboard.tsx` - Add nav buttons
- `dashboard/src/components/Coinpage.tsx` - Auto-save signals to Supabase

## Files Created
- `backtesting_engine.py` - Core backtesting logic
- `dashboard/src/components/Backtest.tsx` + `.css`
- `dashboard/src/components/SignalHistory.tsx` + `.css`
- `migrations/supabase_migration_signals.sql`
