import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { User, Session, AuthError } from '@supabase/supabase-js';
import { supabase, UserProfile, PortfolioHolding, PortfolioPnL, HoldingSnapshot, db } from '../lib/supabase';

// Inactivity timeout in milliseconds (5 minutes)
const INACTIVITY_TIMEOUT = 5 * 60 * 1000;

interface AuthContextType {
  user: User | null;
  profile: UserProfile | null;
  session: Session | null;
  loading: boolean;
  holdings: PortfolioHolding[];
  signUp: (
    email: string,
    password: string,
    username: string
  ) => Promise<{ error: AuthError | null }>;
  signIn: (
    email: string,
    password: string
  ) => Promise<{ error: AuthError | null }>;
  signOut: () => Promise<void>;
  updateProfile: (updates: Partial<UserProfile>) => Promise<boolean>;
  refreshProfile: () => Promise<void>;
  refreshHoldings: () => Promise<void>;
  addHolding: (coin: string, amount: number, price: number) => Promise<boolean>;
  reduceHolding: (coin: string, amount: number) => Promise<boolean>;
  updateCapital: (newCapital: number) => Promise<boolean>;
  recordSnapshot: (totalValue: number, holdingsSnapshot: HoldingSnapshot[]) => Promise<boolean>;
  getPortfolioPnL: (days: number) => Promise<PortfolioPnL | null>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [holdings, setHoldings] = useState<PortfolioHolding[]>([]);
  const [loading, setLoading] = useState(true);

  const inactivityTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // 🔹 SAFE profile fetch (never blocks auth)
  const fetchProfile = async (userId: string) => {
    try {
      const profileData = await db.getProfile(userId);
      setProfile(profileData ?? null);
    } catch (err) {
      console.error('Profile fetch failed:', err);
      setProfile(null);
    }
  };

  // 🔹 Fetch portfolio holdings
  const fetchHoldings = useCallback(async (userId: string) => {
    try {
      const holdingsData = await db.getPortfolioHoldings(userId);
      setHoldings(holdingsData);
    } catch (err) {
      console.error('Holdings fetch failed:', err);
      setHoldings([]);
    }
  }, []);

  // 🔹 Init auth + listener
  useEffect(() => {
    let mounted = true;

    // Check if this is a fresh app start (new tab/refresh)
    const isAppInitialized = sessionStorage.getItem('app_initialized');

    const initAuth = async () => {
      // If fresh start, sign out first to force login
      if (!isAppInitialized) {
        sessionStorage.setItem('app_initialized', 'true');
        await supabase.auth.signOut();
        if (mounted) {
          setSession(null);
          setUser(null);
          setProfile(null);
          setHoldings([]);
          setLoading(false);
        }
        return;
      }

      // Otherwise, check for existing session
      const { data } = await supabase.auth.getSession();
      if (!mounted) return;

      setSession(data.session);
      setUser(data.session?.user ?? null);
      setLoading(false); // ✅ AUTH DONE

      if (data.session?.user) {
        fetchProfile(data.session.user.id); // async, non-blocking
        fetchHoldings(data.session.user.id);
      }
    };

    initAuth();

    // 2️⃣ Auth state changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!mounted) return;

      setSession(session);
      setUser(session?.user ?? null);
      setLoading(false); // ✅ ALWAYS stop loader

      if (session?.user) {
        fetchProfile(session.user.id);
        fetchHoldings(session.user.id);
      } else {
        setProfile(null);
        setHoldings([]);
      }
    });

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, [fetchHoldings]);

  // 🔹 Inactivity auto-logout
  useEffect(() => {
    // Only track activity when user is logged in
    if (!user) {
      if (inactivityTimeoutRef.current) {
        clearTimeout(inactivityTimeoutRef.current);
        inactivityTimeoutRef.current = null;
      }
      return;
    }

    const resetInactivityTimer = () => {
      // Clear existing timeout
      if (inactivityTimeoutRef.current) {
        clearTimeout(inactivityTimeoutRef.current);
      }

      // Set new timeout
      inactivityTimeoutRef.current = setTimeout(async () => {
        console.log('⏰ Inactivity timeout - logging out');
        await supabase.auth.signOut();
        setUser(null);
        setSession(null);
        setProfile(null);
        setHoldings([]);
        // Clear session storage to force login on next visit
        sessionStorage.removeItem('app_initialized');
      }, INACTIVITY_TIMEOUT);
    };

    // Activity events to track
    const activityEvents = ['mousedown', 'mousemove', 'keydown', 'scroll', 'touchstart', 'click'];

    // Add event listeners
    activityEvents.forEach(event => {
      document.addEventListener(event, resetInactivityTimer, { passive: true });
    });

    // Start the timer
    resetInactivityTimer();

    // Cleanup
    return () => {
      activityEvents.forEach(event => {
        document.removeEventListener(event, resetInactivityTimer);
      });
      if (inactivityTimeoutRef.current) {
        clearTimeout(inactivityTimeoutRef.current);
      }
    };
  }, [user]);

  // 🔹 Sign up
  const signUp = async (
    email: string,
    password: string,
    username: string
  ) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          username,
          display_name: username,
        },
      },
    });

    if (!error && data.user) {
      const { error: profileError } = await supabase.rpc(
        'create_profile_for_user',
        {
          p_user_id: data.user.id,
          p_user_email: email,
          p_user_username: username,
        }
      );

      if (profileError) {
        console.error('Profile creation error:', profileError);
        return { error: { message: profileError.message } as AuthError };
      }
    }

    return { error };
  };

  // 🔹 Sign in
  const signIn = async (email: string, password: string) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (!error && data.session?.user) {
      setSession(data.session);
      setUser(data.session.user);
      fetchProfile(data.session.user.id);
    }

    return { error };
  };

  // 🔹 Sign out
  const signOut = async () => {
    await supabase.auth.signOut();
    setUser(null);
    setSession(null);
    setProfile(null);
    setHoldings([]);
  };

  // 🔹 Update profile
  const updateProfile = async (updates: Partial<UserProfile>) => {
    if (!user) return false;

    const success = await db.updateProfile(user.id, updates);
    if (success) {
      fetchProfile(user.id);
    }
    return success;
  };

  // 🔹 Refresh profile
  const refreshProfile = async () => {
    if (user) {
      fetchProfile(user.id);
    }
  };

  // 🔹 Refresh holdings
  const refreshHoldings = async () => {
    if (user) {
      await fetchHoldings(user.id);
    }
  };

  // 🔹 Add holding (buy coins)
  const addHolding = async (coin: string, amount: number, price: number): Promise<boolean> => {
    if (!user) return false;
    const result = await db.addHolding(user.id, coin, amount, price);
    if (result) {
      await fetchHoldings(user.id);
      return true;
    }
    return false;
  };

  // 🔹 Reduce holding (sell coins)
  const reduceHolding = async (coin: string, amount: number): Promise<boolean> => {
    if (!user) return false;
    const result = await db.reduceHolding(user.id, coin, amount);
    if (result) {
      await fetchHoldings(user.id);
      return true;
    }
    return false;
  };

  // 🔹 Update capital
  const updateCapital = async (newCapital: number): Promise<boolean> => {
    if (!user) return false;
    const success = await db.updateCapital(user.id, newCapital);
    if (success) {
      await fetchProfile(user.id);
      return true;
    }
    return false;
  };

  // 🔹 Record daily snapshot
  const recordSnapshot = async (totalValue: number, holdingsSnapshot: HoldingSnapshot[]): Promise<boolean> => {
    if (!user) return false;
    const result = await db.recordSnapshot(user.id, totalValue, holdingsSnapshot);
    return result !== null;
  };

  // 🔹 Get portfolio P&L for time period
  const getPortfolioPnL = async (days: number): Promise<PortfolioPnL | null> => {
    if (!user) return null;
    return await db.getPortfolioPnL(user.id, days);
  };

  const value: AuthContextType = {
    user,
    session,
    profile,
    loading,
    holdings,
    signUp,
    signIn,
    signOut,
    updateProfile,
    refreshProfile,
    refreshHoldings,
    addHolding,
    reduceHolding,
    updateCapital,
    recordSnapshot,
    getPortfolioPnL,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
