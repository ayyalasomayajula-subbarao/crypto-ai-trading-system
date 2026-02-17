import { createClient } from '@supabase/supabase-js';

// Get these from your Supabase project settings -> API
// https://app.supabase.com/project/_/settings/api
const supabaseUrl = process.env.REACT_APP_SUPABASE_URL || '';
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY || '';

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn(
    '⚠️ Supabase credentials not found. Please set REACT_APP_SUPABASE_URL and REACT_APP_SUPABASE_ANON_KEY in your .env file'
  );
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Database types for TypeScript
export interface UserProfile {
  id: string;
  email: string;
  username: string;
  display_name: string | null;
  avatar_url: string | null;
  capital: number;
  experience_level: 'BEGINNER' | 'INTERMEDIATE' | 'ADVANCED';
  default_trade_type: 'SCALP' | 'SHORT_TERM' | 'SWING' | 'INVESTMENT';
  created_at: string;
  updated_at: string;
}

export interface Trade {
  id: string;
  user_id: string;
  coin: string;
  trade_type: string;
  entry_price: number;
  exit_price: number | null;
  amount: number;
  direction: 'LONG' | 'SHORT';
  status: 'OPEN' | 'CLOSED' | 'CANCELLED';
  pnl: number | null;
  pnl_percent: number | null;
  notes: string | null;
  created_at: string;
  closed_at: string | null;
}

export interface Watchlist {
  id: string;
  user_id: string;
  coin: string;
  alerts_enabled: boolean;
  target_price: number | null;
  created_at: string;
}

export interface PortfolioHolding {
  id: string;
  user_id: string;
  coin: string;
  amount: number;
  avg_price: number;
  total_invested: number;
  created_at: string;
  updated_at: string;
}

export interface PortfolioSnapshot {
  id: string;
  user_id: string;
  snapshot_date: string;
  total_value: number;
  total_invested: number;
  capital: number;
  holdings_snapshot: HoldingSnapshot[];
  created_at: string;
}

export interface HoldingSnapshot {
  coin: string;
  amount: number;
  price: number;
  value: number;
}

export interface PortfolioPnL {
  start_value: number;
  end_value: number;
  pnl_amount: number;
  pnl_percent: number;
  period_start: string;
  period_end: string;
}

export interface TradeSignal {
  id: string;
  user_id: string | null;
  coin: string;
  trade_type: string;
  verdict: string;
  confidence: string | null;
  win_probability: number | null;
  loss_probability: number | null;
  sideways_probability: number | null;
  expectancy: number | null;
  price_at_signal: number | null;
  market_regime: string | null;
  reasoning: Record<string, any> | null;
  created_at: string;
}

export interface SignalStats {
  coin: string;
  total: number;
  buy: number;
  wait: number;
  avoid: number;
}

// Helper functions for database operations
export const db = {
  // User Profile
  async getProfile(userId: string): Promise<UserProfile | null> {
    const { data, error } = await supabase
      .from('profiles')
      .select('*')
      .eq('id', userId)
      .single();

    if (error) {
      console.error('Error fetching profile:', error);
      return null;
    }
    return data;
  },

  async updateProfile(userId: string, updates: Partial<UserProfile>): Promise<boolean> {
    const { error } = await supabase
      .from('profiles')
      .update({ ...updates, updated_at: new Date().toISOString() })
      .eq('id', userId);

    if (error) {
      console.error('Error updating profile:', error);
      return false;
    }
    return true;
  },

  // Trades
  async getTrades(userId: string, limit = 50): Promise<Trade[]> {
    const { data, error } = await supabase
      .from('trades')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false })
      .limit(limit);

    if (error) {
      console.error('Error fetching trades:', error);
      return [];
    }
    return data || [];
  },

  async createTrade(trade: Omit<Trade, 'id' | 'created_at'>): Promise<Trade | null> {
    const { data, error } = await supabase
      .from('trades')
      .insert(trade)
      .select()
      .single();

    if (error) {
      console.error('Error creating trade:', error);
      return null;
    }
    return data;
  },

  async updateTrade(tradeId: string, updates: Partial<Trade>): Promise<boolean> {
    const { error } = await supabase
      .from('trades')
      .update(updates)
      .eq('id', tradeId);

    if (error) {
      console.error('Error updating trade:', error);
      return false;
    }
    return true;
  },

  // Watchlist
  async getWatchlist(userId: string): Promise<Watchlist[]> {
    const { data, error } = await supabase
      .from('watchlist')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Error fetching watchlist:', error);
      return [];
    }
    return data || [];
  },

  async addToWatchlist(item: Omit<Watchlist, 'id' | 'created_at'>): Promise<Watchlist | null> {
    const { data, error } = await supabase
      .from('watchlist')
      .insert(item)
      .select()
      .single();

    if (error) {
      console.error('Error adding to watchlist:', error);
      return null;
    }
    return data;
  },

  async removeFromWatchlist(itemId: string): Promise<boolean> {
    const { error } = await supabase
      .from('watchlist')
      .delete()
      .eq('id', itemId);

    if (error) {
      console.error('Error removing from watchlist:', error);
      return false;
    }
    return true;
  },

  // Portfolio Holdings
  async getPortfolioHoldings(userId: string): Promise<PortfolioHolding[]> {
    const { data, error } = await supabase
      .from('portfolio_holdings')
      .select('*')
      .eq('user_id', userId)
      .order('total_invested', { ascending: false });

    if (error) {
      console.error('Error fetching portfolio holdings:', error);
      return [];
    }
    return data || [];
  },

  async addHolding(userId: string, coin: string, amount: number, price: number): Promise<PortfolioHolding | null> {
    const { data, error } = await supabase.rpc('upsert_holding', {
      p_user_id: userId,
      p_coin: coin,
      p_amount: amount,
      p_price: price
    });

    if (error) {
      console.error('Error adding holding:', error);
      return null;
    }
    return data;
  },

  async reduceHolding(userId: string, coin: string, amount: number): Promise<PortfolioHolding | null> {
    const { data, error } = await supabase.rpc('reduce_holding', {
      p_user_id: userId,
      p_coin: coin,
      p_amount: amount
    });

    if (error) {
      console.error('Error reducing holding:', error);
      return null;
    }
    return data;
  },

  async deleteHolding(holdingId: string): Promise<boolean> {
    const { error } = await supabase
      .from('portfolio_holdings')
      .delete()
      .eq('id', holdingId);

    if (error) {
      console.error('Error deleting holding:', error);
      return false;
    }
    return true;
  },

  // Portfolio Snapshots
  async recordSnapshot(
    userId: string,
    totalValue: number,
    holdingsSnapshot: HoldingSnapshot[]
  ): Promise<PortfolioSnapshot | null> {
    const { data, error } = await supabase.rpc('record_portfolio_snapshot', {
      p_user_id: userId,
      p_total_value: totalValue,
      p_holdings_snapshot: holdingsSnapshot
    });

    if (error) {
      console.error('Error recording snapshot:', error);
      return null;
    }
    return data;
  },

  async getSnapshots(userId: string, days: number = 30): Promise<PortfolioSnapshot[]> {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    const { data, error } = await supabase
      .from('portfolio_snapshots')
      .select('*')
      .eq('user_id', userId)
      .gte('snapshot_date', startDate.toISOString().split('T')[0])
      .order('snapshot_date', { ascending: false });

    if (error) {
      console.error('Error fetching snapshots:', error);
      return [];
    }
    return data || [];
  },

  async getPortfolioPnL(userId: string, days: number): Promise<PortfolioPnL | null> {
    const { data, error } = await supabase.rpc('get_portfolio_pnl', {
      p_user_id: userId,
      p_days: days
    });

    if (error) {
      console.error('Error getting portfolio PnL:', error);
      return null;
    }
    return data && data.length > 0 ? data[0] : null;
  },

  // Update user capital
  async updateCapital(userId: string, newCapital: number): Promise<boolean> {
    const { error } = await supabase
      .from('profiles')
      .update({ capital: newCapital, updated_at: new Date().toISOString() })
      .eq('id', userId);

    if (error) {
      console.error('Error updating capital:', error);
      return false;
    }
    return true;
  },

  // Trade Signals (Prediction History)
  async saveSignal(signal: Omit<TradeSignal, 'id' | 'created_at'>): Promise<TradeSignal | null> {
    const { data, error } = await supabase
      .from('trade_signals')
      .insert(signal)
      .select()
      .single();

    if (error) {
      console.error('Error saving signal:', error);
      return null;
    }
    return data;
  },

  async getSignals(userId: string | null, coin?: string, limit = 100): Promise<TradeSignal[]> {
    let query = supabase
      .from('trade_signals')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(limit);

    if (userId) {
      query = query.eq('user_id', userId);
    }
    if (coin) {
      query = query.eq('coin', coin);
    }

    const { data, error } = await query;

    if (error) {
      console.error('Error fetching signals:', error);
      return [];
    }
    return data || [];
  },

  async getSignalStats(userId: string): Promise<SignalStats[]> {
    const { data, error } = await supabase
      .from('trade_signals')
      .select('coin, verdict')
      .eq('user_id', userId);

    if (error) {
      console.error('Error fetching signal stats:', error);
      return [];
    }

    // Aggregate by coin
    const statsMap: Record<string, SignalStats> = {};
    (data || []).forEach((s: { coin: string; verdict: string }) => {
      if (!statsMap[s.coin]) {
        statsMap[s.coin] = { coin: s.coin, total: 0, buy: 0, wait: 0, avoid: 0 };
      }
      statsMap[s.coin].total++;
      if (s.verdict === 'BUY') statsMap[s.coin].buy++;
      else if (s.verdict === 'WAIT') statsMap[s.coin].wait++;
      else statsMap[s.coin].avoid++;
    });

    return Object.values(statsMap);
  }
};
