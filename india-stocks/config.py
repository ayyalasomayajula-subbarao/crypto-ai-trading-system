"""
India Stocks Trading System — Central Configuration
All instrument definitions, constants, TP/SL, feature lists.
"""

# ─── Instruments ────────────────────────────────────────────────────────────

INSTRUMENTS = {
    # ── Indices ──────────────────────────────────────────────────────────────
    "NIFTY50": {
        "yf_symbol":          "^NSEI",
        "nse_symbol":         "NIFTY",
        "display_name":       "NIFTY 50",
        "type":               "index",
        "sector":             "Index",
        "lot_size":           50,
        "tp_pct":             3.5,   # was 2.5 — too tight, frequently SL'd before TP in choppy markets
        "sl_pct":             1.5,   # was 1.0
        "time_limit_days":    7,     # was 5
        "direction_threshold": 1.5,
        "meta_win_thresh":    0.45,  # was global 0.52 — meta over-filters recent folds to 0-3 trades
        "data_start":         "2010-01-01",
        "wf_folds":           6,
        "regime_col":         "1w_dist_sma_50",
    },
    "BANKNIFTY": {
        "yf_symbol":          "^NSEBANK",
        "nse_symbol":         "BANKNIFTY",
        "display_name":       "Bank NIFTY",
        "type":               "index",
        "sector":             "Banking",
        "lot_size":           30,
        "tp_pct":             3.5,
        "sl_pct":             1.5,
        "time_limit_days":    5,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           6,
        "regime_col":         "1w_dist_sma_50",
    },
    "NIFTYIT": {
        "yf_symbol":          "^CNXIT",
        "nse_symbol":         "NIFTYIT",
        "display_name":       "NIFTY IT",
        "type":               "index",
        "sector":             "Technology",
        "lot_size":           30,
        "tp_pct":             4.0,
        "sl_pct":             1.5,
        "time_limit_days":    7,
        "direction_threshold": 2.0,
        "data_start":         "2012-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },

    # ── Top F&O Stocks ───────────────────────────────────────────────────────
    # data_start "2010-01-01" + wf_folds=5: gives 16yr history, ~5 test windows
    # covering pre-2015, 2016-18, COVID 2020, 2021-22 bull/bear, 2023-26 current.
    "RELIANCE": {
        "yf_symbol":          "RELIANCE.NS",
        "nse_symbol":         "RELIANCE",
        "display_name":       "Reliance Industries",
        "type":               "stock",
        "sector":             "Energy",
        "lot_size":           250,
        "tp_pct":             4.5,
        "sl_pct":             1.5,
        "time_limit_days":    12,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "TCS": {
        "yf_symbol":          "TCS.NS",
        "nse_symbol":         "TCS",
        "display_name":       "Tata Consultancy Services",
        "type":               "stock",
        "sector":             "Technology",
        "lot_size":           150,
        "tp_pct":             4.5,
        "sl_pct":             1.5,
        "time_limit_days":    12,
        "direction_threshold": 2.0,
        "meta_win_thresh":    0.0,   # bypass meta — regime shift kills later folds even at 0.45
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "INFY": {
        "yf_symbol":          "INFY.NS",
        "nse_symbol":         "INFY",
        "display_name":       "Infosys",
        "type":               "stock",
        "sector":             "Technology",
        "lot_size":           300,
        "tp_pct":             4.5,
        "sl_pct":             1.5,
        "time_limit_days":    12,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "HDFCBANK": {
        "yf_symbol":          "HDFCBANK.NS",
        "nse_symbol":         "HDFCBANK",
        "display_name":       "HDFC Bank",
        "type":               "stock",
        "sector":             "Banking",
        "lot_size":           550,
        "tp_pct":             3.5,
        "sl_pct":             1.5,
        "time_limit_days":    10,
        "direction_threshold": 1.5,
        "meta_win_thresh":    0.0,   # bypass meta — meta kills folds even at 0.45
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "ICICIBANK": {
        "yf_symbol":          "ICICIBANK.NS",
        "nse_symbol":         "ICICIBANK",
        "display_name":       "ICICI Bank",
        "type":               "stock",
        "sector":             "Banking",
        "lot_size":           700,
        "tp_pct":             4.0,
        "sl_pct":             1.5,
        "time_limit_days":    10,
        "direction_threshold": 1.5,
        "meta_win_thresh":    0.0,   # bypass meta — meta kills folds even at 0.45
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "BHARTIARTL": {
        "yf_symbol":          "BHARTIARTL.NS",
        "nse_symbol":         "BHARTIARTL",
        "display_name":       "Bharti Airtel",
        "type":               "stock",
        "sector":             "Telecom",
        "lot_size":           475,
        "tp_pct":             4.5,
        "sl_pct":             1.5,
        "time_limit_days":    12,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "AXISBANK": {
        "yf_symbol":          "AXISBANK.NS",
        "nse_symbol":         "AXISBANK",
        "display_name":       "Axis Bank",
        "type":               "stock",
        "sector":             "Banking",
        "lot_size":           625,
        "tp_pct":             6.0,
        "sl_pct":             2.0,
        "time_limit_days":    12,
        "direction_threshold": 2.5,
        "meta_win_thresh":    0.45,  # keep existing — gave MARGINAL with 2 valid folds
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "SBIN": {
        "yf_symbol":          "SBIN.NS",
        "nse_symbol":         "SBIN",
        "display_name":       "State Bank of India",
        "type":               "stock",
        "sector":             "Banking",
        "lot_size":           1500,
        "tp_pct":             6.0,
        "sl_pct":             2.0,
        "time_limit_days":    12,
        "direction_threshold": 2.5,
        "meta_win_thresh":    0.0,   # bypass meta — meta kills fold2/3 leaving only 2 valid folds
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "BAJFINANCE": {
        "yf_symbol":          "BAJFINANCE.NS",
        "nse_symbol":         "BAJFINANCE",
        "display_name":       "Bajaj Finance",
        "type":               "stock",
        "sector":             "Finance",
        "lot_size":           125,
        "tp_pct":             6.0,
        "sl_pct":             2.0,
        "time_limit_days":    12,
        "direction_threshold": 2.5,
        "meta_win_thresh":    0.0,   # bypass meta — ADX gate kills fold3 anyway
        "data_start":         "2015-01-01",  # F&O liquidity only from 2014; keep 2015
        "wf_folds":           4,
        "regime_col":         "1w_dist_sma_50",
    },
    "WIPRO": {
        "yf_symbol":          "WIPRO.NS",
        "nse_symbol":         "WIPRO",
        "display_name":       "Wipro",
        "type":               "stock",
        "sector":             "Technology",
        "lot_size":           1500,
        "tp_pct":             4.5,
        "sl_pct":             1.5,
        "time_limit_days":    12,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "HCLTECH": {
        "yf_symbol":          "HCLTECH.NS",
        "nse_symbol":         "HCLTECH",
        "display_name":       "HCL Technologies",
        "type":               "stock",
        "sector":             "Technology",
        "lot_size":           700,
        "tp_pct":             4.5,
        "sl_pct":             1.5,
        "time_limit_days":    12,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "MARUTI": {
        "yf_symbol":          "MARUTI.NS",
        "nse_symbol":         "MARUTI",
        "display_name":       "Maruti Suzuki",
        "type":               "stock",
        "sector":             "Auto",
        "lot_size":           100,
        "tp_pct":             6.0,
        "sl_pct":             2.0,
        "time_limit_days":    12,
        "direction_threshold": 2.5,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "TITAN": {
        "yf_symbol":          "TITAN.NS",
        "nse_symbol":         "TITAN",
        "display_name":       "Titan Company",
        "type":               "stock",
        "sector":             "Consumer",
        "lot_size":           175,
        "tp_pct":             6.0,
        "sl_pct":             2.0,
        "time_limit_days":    12,
        "direction_threshold": 2.5,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "SUNPHARMA": {
        "yf_symbol":          "SUNPHARMA.NS",
        "nse_symbol":         "SUNPHARMA",
        "display_name":       "Sun Pharmaceutical",
        "type":               "stock",
        "sector":             "Pharma",
        "lot_size":           350,
        "tp_pct":             4.5,
        "sl_pct":             1.5,
        "time_limit_days":    12,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },

    # ── New Instruments: Popular F&O Stocks ─────────────────────────────────
    "KOTAKBANK": {
        "yf_symbol":          "KOTAKBANK.NS",
        "nse_symbol":         "KOTAKBANK",
        "display_name":       "Kotak Mahindra Bank",
        "type":               "stock",
        "sector":             "Banking",
        "lot_size":           40,
        "tp_pct":             4.5,
        "sl_pct":             1.5,
        "time_limit_days":    10,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "LT": {
        "yf_symbol":          "LT.NS",
        "nse_symbol":         "LT",
        "display_name":       "Larsen & Toubro",
        "type":               "stock",
        "sector":             "Engineering",
        "lot_size":           25,
        "tp_pct":             5.0,
        "sl_pct":             2.0,
        "time_limit_days":    12,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "BAJAJFINSV": {
        "yf_symbol":          "BAJAJFINSV.NS",
        "nse_symbol":         "BAJAJFINSV",
        "display_name":       "Bajaj Finserv",
        "type":               "stock",
        "sector":             "Finance",
        "lot_size":           50,
        "tp_pct":             5.0,
        "sl_pct":             2.0,
        "time_limit_days":    12,
        "direction_threshold": 2.5,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "DRREDDY": {
        "yf_symbol":          "DRREDDY.NS",
        "nse_symbol":         "DRREDDY",
        "display_name":       "Dr. Reddy's Laboratories",
        "type":               "stock",
        "sector":             "Pharma",
        "lot_size":           25,
        "tp_pct":             5.0,
        "sl_pct":             1.5,
        "time_limit_days":    12,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "ASIANPAINT": {
        "yf_symbol":          "ASIANPAINT.NS",
        "nse_symbol":         "ASIANPAINT",
        "display_name":       "Asian Paints",
        "type":               "stock",
        "sector":             "Consumer",
        "lot_size":           25,
        "tp_pct":             4.5,
        "sl_pct":             1.5,
        "time_limit_days":    10,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "TECHM": {
        "yf_symbol":          "TECHM.NS",
        "nse_symbol":         "TECHM",
        "display_name":       "Tech Mahindra",
        "type":               "stock",
        "sector":             "Technology",
        "lot_size":           75,
        "tp_pct":             5.0,
        "sl_pct":             1.5,
        "time_limit_days":    12,
        "direction_threshold": 2.5,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "ULTRACEMCO": {
        "yf_symbol":          "ULTRACEMCO.NS",
        "nse_symbol":         "ULTRACEMCO",
        "display_name":       "UltraTech Cement",
        "type":               "stock",
        "sector":             "Industrial",
        "lot_size":           25,
        "tp_pct":             5.0,
        "sl_pct":             2.0,
        "time_limit_days":    12,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
    "CIPLA": {
        "yf_symbol":          "CIPLA.NS",
        "nse_symbol":         "CIPLA",
        "display_name":       "Cipla",
        "type":               "stock",
        "sector":             "Pharma",
        "lot_size":           650,
        "tp_pct":             4.5,
        "sl_pct":             1.5,
        "time_limit_days":    12,
        "direction_threshold": 2.0,
        "data_start":         "2010-01-01",
        "wf_folds":           5,
        "regime_col":         "1w_dist_sma_50",
    },
}

# Ordered lists for different use-cases
INDEX_SYMBOLS      = ["NIFTY50", "BANKNIFTY", "NIFTYIT"]
STOCK_SYMBOLS      = [s for s, v in INSTRUMENTS.items() if v["type"] == "stock"]
ALL_SYMBOLS        = list(INSTRUMENTS.keys())

# ACTIVE_SYMBOLS — set via env var on EC2 to restrict to viable/marginal only.
# e.g.  export ACTIVE_SYMBOLS="NIFTY50,BANKNIFTY,NIFTYIT,TITAN,AXISBANK"
# Locally unset → all symbols are active.
import os as _os
_active_env = _os.environ.get("ACTIVE_SYMBOLS", "")
ACTIVE_SYMBOLS = [s.strip() for s in _active_env.split(",") if s.strip() in INSTRUMENTS] \
                 if _active_env else ALL_SYMBOLS

SCAN_SYMBOLS = ACTIVE_SYMBOLS  # symbols included in daily signal scan

# ─── Trading Calendar ───────────────────────────────────────────────────────

MARKET_OPEN_TIME  = "09:15"   # IST
MARKET_CLOSE_TIME = "15:30"   # IST
PRE_OPEN_START    = "09:00"
POST_CLOSE_END    = "16:00"
TIMEZONE          = "Asia/Kolkata"

# NSE holidays 2025-2026 (add more as declared)
NSE_HOLIDAYS = [
    "2025-01-26", "2025-02-19", "2025-03-14", "2025-03-31",
    "2025-04-10", "2025-04-14", "2025-04-18", "2025-05-01",
    "2025-08-15", "2025-08-27", "2025-10-02", "2025-10-02",
    "2025-10-20", "2025-10-23", "2025-11-05", "2025-12-25",
    "2026-01-26", "2026-03-14",
]

# F&O expiry: last Thursday of each month
# Weekly NIFTY/BANKNIFTY: every Thursday

# ─── Model Configuration ────────────────────────────────────────────────────

LGBM_PARAMS = {
    "n_estimators":      700,
    "learning_rate":     0.02,
    "max_depth":         7,
    "num_leaves":        63,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_jobs":            1,         # deterministic
    "random_state":      42,
    "verbose":           -1,
}

META_LGBM_PARAMS = {
    "n_estimators":      200,
    "learning_rate":     0.05,
    "max_depth":         4,
    "num_leaves":        15,
    "min_child_samples": 30,
    "n_jobs":            1,
    "random_state":      42,
    "verbose":           -1,
}

META_LABELING        = True
META_WIN_THRESH      = 0.52
META_TRAIN_FRAC      = 0.25    # last 25% of train window for meta model

# Fast WF params for --fast flag (3-5x quicker sweeps, same viability signal)
WF_FAST_LGBM_PARAMS = {
    "n_estimators":      200,
    "learning_rate":     0.05,
    "max_depth":         6,
    "num_leaves":        31,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "n_jobs":            1,
    "random_state":      42,
    "verbose":           -1,
}

# ─── Walk-Forward Configuration ─────────────────────────────────────────────

WF_THRESHOLD_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                     0.50, 0.55, 0.60, 0.65, 0.70]

VIABLE_SHARPE     = 0.8
VIABLE_WR         = 0.48
MARGINAL_SHARPE   = 0.3
# Daily stock models use 3:1+ R:R (TP 4-6% / SL 1.5-2%) so break-even WR is ~25-27%.
# 38% WR gate gives solid positive EV margin; 44% was designed for 2:1 crypto signals.
MARGINAL_WR       = 0.38
# High-Sharpe secondary MARGINAL path: for systems with excellent risk-adjusted returns
# but lower hit-rate (high R:R structure). At 3:1 R:R, 33% WR still yields ~0.32% EV/trade.
# Requires Sharpe >= 0.70 so single-lucky-year noise is excluded.
MARGINAL_SHARPE_HIGH = 0.70
MARGINAL_WR_LOW      = 0.33
MIN_TRADES        = 15

ADX_GATE          = 20    # only trade when 1d_adx >= 20

# ─── Verdict Thresholds ─────────────────────────────────────────────────────

VERDICT_THRESHOLDS = {
    "STRONG_BUY":   85,
    "BUY":          65,
    "LEAN_BUY":     50,
    "HOLD":         35,
    "LEAN_SELL":    20,
    "SELL":         10,
    "STRONG_SELL":   0,
}

# ─── Signal Weights ─────────────────────────────────────────────────────────

SIGNAL_WEIGHTS = {
    "ml_model":              3.0,
    "price_vs_sma21_1d":     2.0,
    "price_vs_sma50_1d":     2.0,
    "rsi_1d":                1.0,
    "macd_1d":               1.0,
    "adx_1d":                1.5,
    "pcr":                   2.5,
    "india_vix":             2.0,
    "fii_net":               2.5,
    "dii_net":               1.5,
    "oi_change":             1.5,
    "delivery_pct":          1.5,
    "advance_decline":       1.0,
    "gift_nifty":            1.0,
}

# ─── PCR Signal Thresholds ──────────────────────────────────────────────────

PCR_STRONG_BULLISH  = 0.70   # bears overplayed, reversal likely
PCR_BULLISH         = 0.85
PCR_NEUTRAL_LOW     = 0.95
PCR_NEUTRAL_HIGH    = 1.10
PCR_BEARISH         = 1.25
PCR_STRONG_BEARISH  = 1.40

# ─── India VIX Thresholds ───────────────────────────────────────────────────

VIX_EXTREME_FEAR    = 25
VIX_HIGH_FEAR       = 20
VIX_NEUTRAL_HIGH    = 17
VIX_NEUTRAL_LOW     = 13
VIX_COMPLACENCY     = 11

# ─── File Paths ─────────────────────────────────────────────────────────────

import os
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "data")
MODELS_DIR       = os.path.join(BASE_DIR, "models")
LOGS_DIR         = os.path.join(BASE_DIR, "logs")

def ohlcv_path(symbol: str, tf: str) -> str:
    return os.path.join(DATA_DIR, f"{symbol}_{tf}.csv")

def features_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"{symbol}_features.csv")

def model_path(symbol: str, name: str = "wf_decision_model_v2.pkl") -> str:
    return os.path.join(MODELS_DIR, symbol, name)

def features_list_path(symbol: str) -> str:
    return os.path.join(MODELS_DIR, symbol, "decision_features_v2.txt")

FII_DII_PATH         = os.path.join(DATA_DIR, "fii_dii_daily.csv")
INDIA_VIX_PATH       = os.path.join(DATA_DIR, "india_vix.csv")
OPTION_CHAIN_DIR     = os.path.join(DATA_DIR, "option_chain")
EARNINGS_CAL_PATH    = os.path.join(DATA_DIR, "earnings_calendar.csv")
RBI_POLICY_PATH      = os.path.join(DATA_DIR, "rbi_policy_dates.csv")
PAPER_STATE_PATH     = os.path.join(DATA_DIR, "paper_trading_state.json")
AGENT_DB_PATH        = os.path.join(DATA_DIR, "agent_memory.db")

# ─── Sector Index Map ────────────────────────────────────────────────────────

SECTOR_INDEX = {
    "Technology":  "^CNXIT",
    "Banking":     "^NSEBANK",
    "Pharma":      "^CNXPHARMA",
    "Auto":        "^CNXAUTO",
    "Energy":      "^CNXENERGY",
    "Finance":     "^NSEBANK",    # ^CNXFINANCE delisted; banking index is best proxy
    "Telecom":     "^CNXSERVICE",
    "Consumer":    "^NSEI",       # NIFTY50 as broad proxy for consumer discretionary
    "Engineering": "^CNXINFRA",   # Infra/engineering index — covers LT and capital goods
    "Industrial":  "^CNXINFRA",   # Cement/industrial — CNXINFRA is the best proxy
}

# ─── Broker Config (populated from .env) ────────────────────────────────────

ANGEL_ONE_CONFIG = {
    "api_key":       "",   # from .env ANGEL_ONE_API_KEY
    "client_id":     "",   # from .env ANGEL_ONE_CLIENT_ID
    "password":      "",   # from .env ANGEL_ONE_PASSWORD
    "totp_secret":   "",   # from .env ANGEL_ONE_TOTP_SECRET
}
