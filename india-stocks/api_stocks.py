"""
India Stocks API — FastAPI backend on port 8001
All endpoints for stock data, verdicts, F&O, FII/DII, paper trading, live trading.

Run: uvicorn api_stocks:app --port 8001 --reload
"""

import os
import sys
import json
import time
import logging
import asyncio
from datetime import datetime
from typing import Optional

import pytz
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INSTRUMENTS, ALL_SYMBOLS, ACTIVE_SYMBOLS, INDEX_SYMBOLS, STOCK_SYMBOLS,
    DATA_DIR, MODELS_DIR, TIMEZONE, FII_DII_PATH, INDIA_VIX_PATH,
    OPTION_CHAIN_DIR,
)
from precision_verdict import VerdictEngine
from paper_trader import PaperTrader, is_market_open, get_wf_multiplier, get_wf_tier
from broker_angel_one import AngelOneBroker

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IST      = pytz.timezone(TIMEZONE)
app      = FastAPI(title="India Stocks AI API", version="1.0.0")
verdict  = VerdictEngine()
paper    = PaperTrader()
broker   = AngelOneBroker(live=False)  # paper mode by default

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── WebSocket price broadcaster ─────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()

# ─── Paper trading activity log ───────────────────────────────────────────────
# Ring buffer of last 100 scanner events (signal checks, entries, exits, skips)
_pt_activity: list[dict] = []
_PT_MAX_EVENTS = 100

def _pt_log(event: str, symbol: str = "", detail: str = "", verdict: str = ""):
    """Append a paper trading scanner event to the activity log."""
    _pt_activity.append({
        "ts":      datetime.now(IST).strftime("%H:%M:%S"),
        "date":    datetime.now(IST).strftime("%Y-%m-%d"),
        "event":   event,   # SCAN_START | SIGNAL | ENTRY | SKIP | EXIT | ERROR
        "symbol":  symbol,
        "verdict": verdict,
        "detail":  detail,
    })
    if len(_pt_activity) > _PT_MAX_EVENTS:
        _pt_activity.pop(0)

# ─── API-level response cache ─────────────────────────────────────────────────

_api_cache: dict = {}
_api_cache_ts: dict = {}

def _cache_get(key: str, ttl: float):
    if key in _api_cache and time.time() - _api_cache_ts.get(key, 0) < ttl:
        return _api_cache[key]
    return None

def _cache_set(key: str, value):
    _api_cache[key] = value
    _api_cache_ts[key] = time.time()


# ─── Background price loop ────────────────────────────────────────────────────

async def price_broadcast_loop():
    """Stream live prices via WebSocket every 5 seconds."""
    while True:
        try:
            prices = await _get_live_prices()
            paper.update_prices({s: p["price"] for s, p in prices.items()})
            await ws_manager.broadcast({"type": "prices", "data": prices})
        except Exception as e:
            log.warning(f"Price broadcast error: {e}")
        await asyncio.sleep(5)


async def _warm_caches():
    """Pre-warm scan + prices caches so first user request is instant."""
    await asyncio.sleep(1)  # let startup finish
    log.info("Warming caches in background...")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, verdict.scan_all)
        _cache_set("scan", result)
        log.info(f"Scan cache ready: {len(result)} symbols")
    except Exception as e:
        log.warning(f"Scan warm-up failed: {e}")
    try:
        prices = await _get_live_prices()
        log.info(f"Prices cache ready: {len(prices)} symbols")
    except Exception as e:
        log.warning(f"Prices warm-up failed: {e}")


def _paper_scanner_tick():
    """
    One scanner tick: scan all symbols, enter positions for strong signals.
    Runs in a thread executor (blocking). Called every 15 min during market hours.
    """
    if not paper._state.get("running", False):
        return
    if not is_market_open():
        return

    _pt_log("SCAN_START", detail=f"Scanning {len(ACTIVE_SYMBOLS)} symbols")
    log.info("[PaperScanner] Scanning all symbols...")

    try:
        signals = verdict.scan_all()
    except Exception as e:
        _pt_log("ERROR", detail=f"scan_all failed: {e}")
        log.warning(f"[PaperScanner] scan_all error: {e}")
        return

    # Bidirectional filter — match how WF backtest selected trades:
    #   LONG:  direction=LONG  + score >= 55  (BUY / STRONG_BUY range)
    #   SHORT: direction=SHORT + score <= 40  (SELL / STRONG_SELL range, bearish = low score)
    actionable = [
        s for s in signals
        if (
            (s.get("direction") == "LONG"  and s.get("score", 0) >= 55) or
            (s.get("direction") == "SHORT" and s.get("score", 0) <= 40)
        )
    ]

    _pt_log("SCAN_START", detail=f"Found {len(actionable)} actionable signals out of {len(signals)}")

    for sig in actionable:
        sym = sig["symbol"]
        v   = sig.get("verdict", "")
        score = sig.get("score", 0)

        # Skip already open
        if any(p["symbol"] == sym for p in paper._state.get("open_positions", [])):
            _pt_log("SKIP", sym, f"Already in position", v)
            continue

        result = paper.open_position(sym, sig)
        if result.get("ok"):
            _pt_log("ENTRY", sym,
                    f"score={score} entry={sig.get('entry_price',0):.2f} "
                    f"tp={result.get('target_price',0):.2f} sl={result.get('sl_price',0):.2f}", v)
            log.info(f"[PaperScanner] ENTERED {sym} {v} score={score}")
        else:
            reason = result.get("reason", "unknown")
            _pt_log("SKIP", sym, reason, v)
            log.info(f"[PaperScanner] SKIP {sym}: {reason}")


async def _paper_scanner_loop():
    """Background loop: run scanner every 15 minutes during market hours."""
    await asyncio.sleep(5)   # wait for startup
    loop = asyncio.get_event_loop()
    while True:
        if is_market_open():
            try:
                await loop.run_in_executor(None, _paper_scanner_tick)
            except Exception as e:
                log.warning(f"[PaperScanner] tick error: {e}")
        await asyncio.sleep(15 * 60)   # 15-minute interval


@app.on_event("startup")
async def startup():
    asyncio.create_task(price_broadcast_loop())
    asyncio.create_task(_warm_caches())
    asyncio.create_task(_paper_scanner_loop())
    log.info("India Stocks API started on port 8001")


# ─── Price helpers ────────────────────────────────────────────────────────────

def _fetch_intraday_prices() -> dict:
    """
    During market hours: fetch live intraday prices via yfinance 1m (~15 min delay).
    Returns dict keyed by symbol with current price, open, high, low, change_pct vs prev close.
    """
    import yfinance as yf
    import pandas as pd
    from config import ohlcv_path

    result = {}
    # Batch download all symbols in one call
    yf_symbols = [INSTRUMENTS[s]["yf_symbol"] for s in ACTIVE_SYMBOLS if s in INSTRUMENTS]
    sym_map    = {INSTRUMENTS[s]["yf_symbol"]: s for s in ACTIVE_SYMBOLS if s in INSTRUMENTS}

    try:
        raw = yf.download(
            tickers=" ".join(yf_symbols),
            period="1d", interval="1m",
            group_by="ticker", auto_adjust=True, progress=False,
        )
    except Exception as e:
        log.warning(f"Intraday yfinance fetch failed: {e}")
        return {}

    for yf_sym, sym in sym_map.items():
        try:
            # Multi-ticker download uses (ticker, field) column MultiIndex
            if len(yf_symbols) > 1:
                df = raw[yf_sym].dropna(how="all")
            else:
                df = raw.dropna(how="all")

            if df.empty:
                continue

            # Prev close from 1D CSV
            prev_close = 0.0
            path_1d = ohlcv_path(sym, "1d")
            if os.path.exists(path_1d):
                hist = pd.read_csv(path_1d, index_col=0, parse_dates=True)
                if not hist.empty:
                    prev_close = float(hist["close"].iloc[-1])

            current = float(df["Close"].iloc[-1])
            day_open = float(df["Open"].iloc[0])
            day_high = float(df["High"].max())
            day_low  = float(df["Low"].min())
            volume   = int(df["Volume"].sum())
            change_pct = ((current - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            result[sym] = {
                "symbol":       sym,
                "price":        round(current, 2),
                "open":         round(day_open, 2),
                "high":         round(day_high, 2),
                "low":          round(day_low, 2),
                "prev_close":   round(prev_close, 2),
                "volume":       volume,
                "change_pct":   round(change_pct, 2),
                "timestamp":    str(df.index[-1]),
                "display_name": INSTRUMENTS[sym].get("display_name", sym),
                "sector":       INSTRUMENTS[sym].get("sector", ""),
                "type":         INSTRUMENTS[sym].get("type", ""),
                "live":         True,
            }
        except Exception as e:
            log.warning(f"Intraday parse {sym}: {e}")

    return result


async def _get_live_prices() -> dict:
    """
    During market hours: live intraday prices (yfinance 1m, ~15 min delay).
    Outside market hours: last stored 1D close.
    Cache: 60s during market, 300s outside.
    """
    market_open = is_market_open()
    ttl = 60 if market_open else 300
    cached = _cache_get("prices", ttl)
    if cached is not None:
        return cached

    import pandas as pd
    from config import ohlcv_path

    prices = {}

    # During market hours — fetch live intraday
    if market_open:
        try:
            prices = await asyncio.get_event_loop().run_in_executor(
                None, _fetch_intraday_prices
            )
        except Exception as e:
            log.warning(f"Intraday fetch error: {e}")

    # Fill any missing symbols (or all symbols outside market hours) from 1D CSV
    for sym in ACTIVE_SYMBOLS:
        if sym in prices:
            continue
        path = ohlcv_path(sym, "1d")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if df.empty:
                continue
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            change_pct = ((last["close"] - prev["close"]) / prev["close"] * 100
                          if prev["close"] > 0 else 0)
            prices[sym] = {
                "symbol":       sym,
                "price":        round(float(last["close"]), 2),
                "open":         round(float(last["open"]), 2),
                "high":         round(float(last["high"]), 2),
                "low":          round(float(last["low"]), 2),
                "prev_close":   round(float(prev["close"]), 2),
                "volume":       int(last["volume"]),
                "change_pct":   round(change_pct, 2),
                "timestamp":    str(df.index[-1]),
                "display_name": INSTRUMENTS[sym].get("display_name", sym),
                "sector":       INSTRUMENTS[sym].get("sector", ""),
                "type":         INSTRUMENTS[sym].get("type", ""),
                "live":         False,
            }
        except Exception:
            pass

    _cache_set("prices", prices)
    return prices


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "India Stocks AI API", "version": "1.0.0",
            "status": "running", "port": 8001}


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "market_open":   is_market_open(),
        "timestamp":     datetime.now(IST).isoformat(),
        "active_symbols": ACTIVE_SYMBOLS,
        "total_symbols": len(ALL_SYMBOLS),
    }


# ─── Instrument info ──────────────────────────────────────────────────────────

@app.get("/stocks/instruments")
def get_instruments():
    return [
        {
            "symbol":        sym,
            "display_name":  cfg["display_name"],
            "type":          cfg["type"],
            "sector":        cfg.get("sector", ""),
            "lot_size":      cfg["lot_size"],
            "tp_pct":        cfg["tp_pct"],
            "sl_pct":        cfg["sl_pct"],
        }
        for sym, cfg in INSTRUMENTS.items()
    ]


@app.get("/stocks/instruments/{symbol}")
def get_instrument(symbol: str):
    cfg = INSTRUMENTS.get(symbol.upper())
    if not cfg:
        raise HTTPException(404, f"Symbol {symbol} not found")
    return cfg


# ─── Prices ──────────────────────────────────────────────────────────────────

@app.get("/stocks/prices")
async def get_prices():
    return await _get_live_prices()


@app.get("/stocks/prices/{symbol}")
async def get_price(symbol: str):
    prices = await _get_live_prices()
    sym = symbol.upper()
    if sym not in prices:
        raise HTTPException(404, f"Price not found for {sym}")
    return prices[sym]


# ─── OHLCV candles ────────────────────────────────────────────────────────────

@app.get("/stocks/klines/{symbol}")
def get_klines(symbol: str, tf: str = "1d", limit: int = 500):
    import pandas as pd
    import yfinance as yf
    from config import ohlcv_path
    sym  = symbol.upper()
    if sym not in INSTRUMENTS:
        raise HTTPException(404, f"Symbol {sym} not found")
    cfg = INSTRUMENTS[sym]

    # Intraday timeframes: fetch directly from yfinance (no local CSV)
    yf_interval_map = {"1h": "1h", "4h": "60m", "1w": "1wk", "1W": "1wk"}
    if tf in yf_interval_map:
        yf_sym   = cfg["yf_symbol"]
        yf_tf    = yf_interval_map[tf]
        period   = "60d" if tf in ("1h", "4h") else "10y"
        try:
            df = yf.download(yf_sym, period=period, interval=yf_tf,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna(subset=["close"]).tail(limit)
            candles = [
                {
                    "time":   int(ts.timestamp()),
                    "open":   round(float(row["open"]), 2),
                    "high":   round(float(row["high"]), 2),
                    "low":    round(float(row["low"]), 2),
                    "close":  round(float(row["close"]), 2),
                    "volume": int(row.get("volume", 0)),
                }
                for ts, row in df.iterrows()
            ]
            return {"symbol": sym, "tf": tf, "candles": candles}
        except Exception as e:
            raise HTTPException(500, f"yfinance error for {sym}/{tf}: {e}")

    # Daily data: read from local CSV (faster, 10yr history)
    path = ohlcv_path(sym, "1d")
    if not os.path.exists(path):
        raise HTTPException(404, f"OHLCV not found for {sym}/1d")

    df = pd.read_csv(path, index_col=0, parse_dates=True).tail(limit)
    return {
        "symbol": sym,
        "tf":     tf,
        "candles": [
            {
                "time":   int(pd.Timestamp(ts).timestamp()),
                "open":   round(float(row["open"]), 2),
                "high":   round(float(row["high"]), 2),
                "low":    round(float(row["low"]), 2),
                "close":  round(float(row["close"]), 2),
                "volume": int(row["volume"]),
            }
            for ts, row in df.iterrows()
        ],
    }


# ─── Verdict ─────────────────────────────────────────────────────────────────

@app.get("/stocks/verdict/{symbol}")
def get_verdict(symbol: str, force: bool = False):
    sym = symbol.upper()
    if sym not in INSTRUMENTS:
        raise HTTPException(404, f"Symbol {sym} not found")
    result = verdict.get_verdict(sym, force=force)
    result["wf_tier"]       = get_wf_tier(sym)
    result["wf_multiplier"] = get_wf_multiplier(sym)
    return result


@app.get("/stocks/verdict-history/{symbol}")
def get_verdict_history(symbol: str, limit: int = 50):
    sym = symbol.upper()
    return verdict.get_verdict_history(sym, limit=limit)


@app.get("/stocks/accuracy/{symbol}")
def get_accuracy(symbol: str, days: int = 30):
    sym = symbol.upper()
    if sym not in INSTRUMENTS:
        raise HTTPException(404, f"Symbol {sym} not found")
    return verdict.get_accuracy_stats(sym, days=days)


@app.get("/stocks/accuracy")
def get_all_accuracy(days: int = 30):
    """Accuracy stats for all viable/marginal symbols."""
    return {sym: verdict.get_accuracy_stats(sym, days=days) for sym in ACTIVE_SYMBOLS}


@app.get("/stocks/scan")
async def scan_all():
    """Scan all symbols and return ranked verdicts."""
    cached = _cache_get("scan", 300)
    if cached is not None:
        return cached
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, verdict.scan_all)
    _cache_set("scan", result)
    return result


@app.post("/stocks/cache/refresh")
async def refresh_cache():
    """Force-refresh the scan + prices cache."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, verdict.scan_all)
    _cache_set("scan", result)
    prices = await _get_live_prices()
    _cache_set("prices", prices)
    return {"ok": True, "symbols_refreshed": len(result)}


# ─── Market data ──────────────────────────────────────────────────────────────

@app.get("/stocks/market/overview")
def market_overview():
    cached = _cache_get("overview", 300)
    if cached is not None:
        return cached

    import pandas as pd

    result = {
        "timestamp":     datetime.now(IST).isoformat(),
        "market_open":   is_market_open(),
        "india_vix":     None,
        "fii_dii":       None,
        "option_chain":  {},
    }

    # India VIX
    if os.path.exists(INDIA_VIX_PATH):
        try:
            vix = pd.read_csv(INDIA_VIX_PATH, index_col=0, parse_dates=True)
            if not vix.empty:
                last_vix = float(vix["india_vix"].iloc[-1])
                prev_vix = float(vix["india_vix"].iloc[-2]) if len(vix) > 1 else last_vix
                result["india_vix"] = {
                    "value":      round(last_vix, 2),
                    "change_pct": round((last_vix - prev_vix) / prev_vix * 100, 2),
                    "date":       str(vix.index[-1].date()),
                    "level":      ("EXTREME_FEAR" if last_vix >= 25 else
                                   "HIGH_FEAR"   if last_vix >= 20 else
                                   "NEUTRAL"     if last_vix >= 13 else
                                   "COMPLACENCY"),
                }
        except Exception as e:
            log.warning(f"VIX overview: {e}")

    # FII/DII
    if os.path.exists(FII_DII_PATH):
        try:
            fii = pd.read_csv(FII_DII_PATH, index_col=0, parse_dates=True)
            if not fii.empty:
                last = fii.iloc[-1]
                result["fii_dii"] = {
                    "date":                 str(fii.index[-1].date()),
                    "fii_net":              round(float(last.get("fii_net_value", 0)), 2),
                    "dii_net":              round(float(last.get("dii_net_value", 0)), 2),
                    "fii_7d_cumulative":    round(float(last.get("fii_7d_cumulative", 0)), 2),
                    "dii_7d_cumulative":    round(float(last.get("dii_7d_cumulative", 0)), 2),
                    "fii_trend":            int(last.get("fii_trend", 0)),
                    "fii_net_label":        "BUYING" if last.get("fii_net_value", 0) > 0 else "SELLING",
                    "dii_net_label":        "BUYING" if last.get("dii_net_value", 0) > 0 else "SELLING",
                }
        except Exception as e:
            log.warning(f"FII overview: {e}")

    # PCR for indices
    metrics_path = os.path.join(OPTION_CHAIN_DIR, "latest_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                oc_data = json.load(f)
            for sym in ["NIFTY50", "BANKNIFTY"]:
                if sym in oc_data:
                    m = oc_data[sym]
                    result["option_chain"][sym] = {
                        "pcr":             m.get("pcr"),
                        "pcr_7d_avg":      m.get("pcr_7d_avg"),
                        "max_pain":        m.get("max_pain"),
                        "max_pain_dist_pct": m.get("max_pain_dist_pct"),
                        "iv_percentile":   m.get("iv_percentile"),
                    }
        except Exception as e:
            log.warning(f"OC metrics: {e}")

    _cache_set("overview", result)
    return result


@app.get("/stocks/fii-dii")
def get_fii_dii(days: int = 30):
    import pandas as pd
    if not os.path.exists(FII_DII_PATH):
        raise HTTPException(404, "FII/DII data not collected yet")

    df = pd.read_csv(FII_DII_PATH, index_col=0, parse_dates=True).tail(days)
    records = []
    for ts, row in df.iterrows():
        records.append({
            "date":           str(ts.date()),
            "fii_net":        round(float(row.get("fii_net_value", 0)), 2),
            "dii_net":        round(float(row.get("dii_net_value", 0)), 2),
            "fii_7d_cumul":   round(float(row.get("fii_7d_cumulative", 0)), 2),
            "dii_7d_cumul":   round(float(row.get("dii_7d_cumulative", 0)), 2),
            "divergence":     round(float(row.get("fii_dii_divergence", 0)), 2),
        })
    return {"days": days, "data": records}


@app.get("/stocks/option-chain/{symbol}")
def get_option_chain(symbol: str):
    sym = symbol.upper()
    from collect_option_chain import update_symbol_oc, load_latest_metrics
    try:
        metrics = load_latest_metrics(sym)
        if not metrics:
            metrics = update_symbol_oc(sym, save_history=True)
        return metrics
    except Exception as e:
        raise HTTPException(500, f"Option chain error: {e}")


@app.get("/stocks/india-vix")
def get_india_vix(days: int = 90):
    import pandas as pd
    if not os.path.exists(INDIA_VIX_PATH):
        raise HTTPException(404, "India VIX data not collected yet")

    df = pd.read_csv(INDIA_VIX_PATH, index_col=0, parse_dates=True).tail(days)
    return {
        "days": days,
        "current": round(float(df["india_vix"].iloc[-1]), 2) if not df.empty else None,
        "data": [
            {"date": str(ts.date()), "vix": round(float(row["india_vix"]), 2)}
            for ts, row in df.iterrows()
        ],
    }


# ─── Backtesting ─────────────────────────────────────────────────────────────

@app.get("/stocks/backtest/{symbol}")
def get_backtest(symbol: str):
    sym = symbol.upper()
    results_path = os.path.join(MODELS_DIR, sym, "wf_results.json")
    if not os.path.exists(results_path):
        raise HTTPException(404, f"No WF results for {sym}. Run walk_forward.py first.")
    with open(results_path) as f:
        return json.load(f)


@app.get("/stocks/backtest")
def get_all_backtests():
    results = {}
    for sym in ACTIVE_SYMBOLS:
        path = os.path.join(MODELS_DIR, sym, "wf_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results[sym] = json.load(f)
    return results


# ─── Paper trading ───────────────────────────────────────────────────────────

@app.get("/stocks/paper-trading/status")
def paper_status():
    return paper.get_status()


@app.get("/stocks/paper-trading/metrics")
def paper_metrics():
    return paper.get_metrics()


@app.get("/stocks/paper-trading/trades")
def paper_trades(limit: int = 100):
    state = paper._state
    return {
        "open":   state["open_positions"],
        "closed": state["closed_trades"][-limit:],
    }


class TradeRequest(BaseModel):
    symbol: str
    force:  bool = False


@app.post("/stocks/paper-trading/open")
def paper_open(req: TradeRequest):
    sym = req.symbol.upper()
    v = verdict.get_verdict(sym, force=req.force)
    return paper.open_position(sym, v)


@app.post("/stocks/paper-trading/close/{symbol}")
def paper_close(symbol: str):
    sym = symbol.upper()
    from config import ohlcv_path
    import pandas as pd
    path = ohlcv_path(sym, "1d")
    price = 0.0
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not df.empty:
            price = float(df["close"].iloc[-1])
    return paper.close_position(sym, price, "MANUAL")


@app.post("/stocks/paper-trading/start")
def paper_start():
    paper.start()
    return {"ok": True, "status": "running"}


@app.post("/stocks/paper-trading/stop")
def paper_stop():
    paper.stop()
    return {"ok": True, "status": "stopped"}


@app.post("/stocks/paper-trading/reset")
def paper_reset():
    paper.reset()
    return {"ok": True, "message": "Paper trading reset"}


@app.get("/stocks/paper-trading/activity")
def paper_activity(limit: int = 50):
    """Return recent paper trading scanner activity log."""
    events = _pt_activity[-limit:][::-1]  # newest first
    return {
        "events": events,
        "total":  len(_pt_activity),
        "market_open": is_market_open(),
        "running": paper._state.get("running", False),
        "next_scan": "Every 15 minutes during market hours (09:15–15:30 IST)",
    }


# ─── Live trading ─────────────────────────────────────────────────────────────

class LiveOrderRequest(BaseModel):
    symbol:     str
    qty:        int
    side:       str       # BUY or SELL
    price:      float
    sl_price:   float
    product:    str = "CARRYFORWARD"


class BrokerConnectRequest(BaseModel):
    api_key:     str
    client_id:   str
    password:    str
    totp_secret: str


@app.post("/stocks/broker/connect")
def broker_connect(req: BrokerConnectRequest):
    """Connect Angel One broker (stores credentials for session)."""
    os.environ["ANGEL_ONE_API_KEY"]     = req.api_key
    os.environ["ANGEL_ONE_CLIENT_ID"]   = req.client_id
    os.environ["ANGEL_ONE_PASSWORD"]    = req.password
    os.environ["ANGEL_ONE_TOTP_SECRET"] = req.totp_secret
    global broker
    broker = AngelOneBroker(live=True)
    return {"ok": True, "message": "Broker connected"}


@app.get("/stocks/broker/status")
def broker_status():
    return {
        "connected": broker._conn is not None,
        "live":      broker.live,
        "funds":     broker.get_funds() if broker.live else None,
    }


@app.post("/stocks/broker/order")
def broker_order(req: LiveOrderRequest):
    if not broker.live:
        raise HTTPException(400, "Broker not connected. POST /stocks/broker/connect first.")
    return broker.place_order(
        req.symbol.upper(), req.qty, req.side,
        req.price, req.sl_price, req.product
    )


@app.get("/stocks/broker/positions")
def broker_positions():
    return {"positions": broker.get_positions()}


@app.post("/stocks/broker/emergency-exit")
def broker_emergency_exit():
    if not broker.live:
        return {"ok": False, "reason": "Broker not connected"}
    results = broker.close_all_positions()
    return {"ok": True, "closed": results}


# ─── Agent endpoints ──────────────────────────────────────────────────────────

@app.get("/stocks/agents/scan")
async def agents_scan():
    cached = _cache_get("scan", 300)
    if cached is not None:
        return cached
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, verdict.scan_all)
    _cache_set("scan", result)
    return result


@app.get("/stocks/agents/report")
def agents_report():
    all_bt = {}
    for sym in ACTIVE_SYMBOLS:
        path = os.path.join(MODELS_DIR, sym, "wf_results.json")
        if os.path.exists(path):
            with open(path) as f:
                all_bt[sym] = json.load(f)

    viable   = [s for s, r in all_bt.items() if r.get("verdict") == "VIABLE"]
    marginal = [s for s, r in all_bt.items() if r.get("verdict") == "MARGINAL"]

    return {
        "timestamp":     datetime.now(IST).isoformat(),
        "viable":        viable,
        "marginal":      marginal,
        "not_viable":    [s for s in ACTIVE_SYMBOLS if s not in viable + marginal],
        "paper_metrics": paper.get_metrics(),
        "market_open":   is_market_open(),
    }


# ─── Data pipeline trigger ───────────────────────────────────────────────────

@app.post("/stocks/update-data")
async def update_data(background_tasks: BackgroundTasks):
    """Trigger data collection in background."""
    def _run():
        import subprocess
        scripts = [
            "collect_nse_data.py",
            "collect_fii_dii.py",
            "collect_option_chain.py --history",
        ]
        for script in scripts:
            try:
                subprocess.run(
                    ["python", os.path.join(os.path.dirname(__file__), script.split()[0])]
                    + script.split()[1:],
                    timeout=300,
                    capture_output=True,
                )
            except Exception as e:
                log.error(f"Data update {script}: {e}")
    background_tasks.add_task(_run)
    return {"ok": True, "message": "Data update triggered in background"}


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/stocks/prices")
async def ws_prices(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Send initial snapshot so client has data immediately on connect
        prices = await _get_live_prices()
        await websocket.send_json({"type": "initial", "data": prices})
        while True:
            await websocket.receive_text()  # keep-alive ping
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_stocks:app", host="0.0.0.0", port=8001, reload=True)
