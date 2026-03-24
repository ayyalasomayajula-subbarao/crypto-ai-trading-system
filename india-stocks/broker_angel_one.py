"""
Angel One SmartAPI Broker Integration (FREE)
Supports: market/limit/SL orders, position management, live quotes.
Requires: pip install smartapi-python pyotp

Setup:
    1. Open Angel One demat account
    2. Create API key at smartapi.angelbroking.com
    3. Set env vars: ANGEL_ONE_API_KEY, ANGEL_ONE_CLIENT_ID,
                     ANGEL_ONE_PASSWORD, ANGEL_ONE_TOTP_SECRET
"""

from __future__ import annotations
import os
import sys
import logging
import time
from datetime import datetime
from typing import Optional

import pytz
import pyotp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INSTRUMENTS, TIMEZONE

log = logging.getLogger(__name__)
IST = pytz.timezone(TIMEZONE)

# ─── Symbol token cache ───────────────────────────────────────────────────────

_TOKEN_CACHE: dict[str, str] = {
    # Indices
    "NIFTY50":    "99926000",
    "BANKNIFTY":  "99926009",
    "NIFTYIT":    "99926009",   # mapped to NIFTYIT index token
    # Stocks
    "RELIANCE":   "2885",
    "TCS":        "11536",
    "INFY":       "1594",
    "HDFCBANK":   "1333",
    "ICICIBANK":  "4963",
    "BHARTIARTL": "10604",
    "AXISBANK":   "5900",
    "SBIN":       "3045",
    "BAJFINANCE": "317",
    "WIPRO":      "3787",
    "HCLTECH":    "7229",
    "MARUTI":     "10999",
    "SUNPHARMA":  "3351",
    "TITAN":      "3506",
    "LT":         "11483",
    "DRREDDY":    "881",
    "BAJAJFINSV": "16675",
    "ULTRACEMCO": "11532",
    "ASIANPAINT": "236",
    "KOTAKBANK":  "1922",
    "CIPLA":      "694",
    "TECHM":      "13538",
}

# ─── Broker class ─────────────────────────────────────────────────────────────

class AngelOneBroker:
    """
    Angel One SmartAPI wrapper.
    Supports paper mode (dry-run) and live mode.
    """

    def __init__(self, live: bool = False):
        self.live    = live
        self.api_key = os.getenv("ANGEL_ONE_API_KEY", "")
        self.client  = os.getenv("ANGEL_ONE_CLIENT_ID", "")
        self.password = os.getenv("ANGEL_ONE_PASSWORD", "")
        self.totp_secret = os.getenv("ANGEL_ONE_TOTP_SECRET", "")
        self._conn = None

        # Always connect if credentials available — needed for live LTP even in paper mode
        if self.api_key and self.client and self.totp_secret:
            self._connect()
        elif live:
            log.error("Live mode requested but ANGEL_ONE_* env vars not set")

    def _connect(self) -> bool:
        """Authenticate and create SmartAPI session."""
        try:
            from SmartApi import SmartConnect
            self._conn = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now() if self.totp_secret else "000000"
            data = self._conn.generateSession(self.client, self.password, totp)
            if data and data.get("status"):
                log.info("Angel One: connected successfully")
                return True
            log.error(f"Angel One auth failed: {data}")
            return False
        except ImportError:
            log.error("smartapi-python not installed: pip install smartapi-python pyotp")
            return False
        except Exception as e:
            log.error(f"Angel One connect error: {e}")
            return False

    def _get_token(self, symbol: str) -> str:
        return _TOKEN_CACHE.get(symbol, "")

    # ── Live quotes ───────────────────────────────────────────────────────────

    # Index trading symbols on Angel One
    _INDEX_TRADING_SYMBOLS: dict[str, str] = {
        "NIFTY50":   "Nifty 50",
        "BANKNIFTY": "Nifty Bank",
        "NIFTYIT":   "Nifty IT",
    }

    def get_ltp(self, symbol: str) -> float:
        """Get Last Traded Price. Works in both paper and live mode if credentials set."""
        if not self._conn:
            return 0.0
        try:
            token = self._get_token(symbol)
            if not token:
                return 0.0
            cfg  = INSTRUMENTS.get(symbol, {})
            if cfg.get("type") == "index":
                trad_sym = self._INDEX_TRADING_SYMBOLS.get(symbol, symbol)
            else:
                trad_sym = f"{symbol}-EQ"
            result = self._conn.ltpData("NSE", trad_sym, token)
            return float(result.get("data", {}).get("ltp", 0))
        except Exception as e:
            log.warning(f"LTP fetch {symbol}: {e}")
            return 0.0

    def get_ltp_batch(self, symbols: list[str]) -> dict[str, float]:
        """Fetch LTP for multiple symbols. Returns {symbol: ltp}."""
        return {sym: self.get_ltp(sym) for sym in symbols}

    def get_quote(self, symbol: str) -> dict:
        """Get full quote for a symbol."""
        if not self.live or not self._conn:
            return {}
        try:
            token  = self._get_token(symbol)
            exch   = "NSE"
            result = self._conn.getMarketData("FULL", exch, token)
            d = result.get("data", {})
            return {
                "symbol":  symbol,
                "ltp":     float(d.get("ltp", 0)),
                "open":    float(d.get("open", 0)),
                "high":    float(d.get("high", 0)),
                "low":     float(d.get("low", 0)),
                "close":   float(d.get("close", 0)),
                "volume":  int(d.get("totTrdVol", 0)),
                "change_pct": float(d.get("percentChange", 0)),
            }
        except Exception as e:
            log.warning(f"Quote fetch {symbol}: {e}")
            return {}

    # ── Order placement ───────────────────────────────────────────────────────

    def place_order(self, symbol: str, qty: int, side: str,
                    price: float, sl_price: float,
                    product: str = "CARRYFORWARD",
                    order_type: str = "LIMIT") -> dict:
        """
        Place an order with Angel One.

        Args:
            symbol:     e.g. "RELIANCE"
            qty:        quantity (must be multiple of lot_size for F&O)
            side:       "BUY" or "SELL"
            price:      limit price (0 for market)
            sl_price:   stop-loss price
            product:    "INTRADAY" or "CARRYFORWARD"
            order_type: "LIMIT", "MARKET", "STOPLOSS_LIMIT"
        """
        if not self.live:
            result = {
                "ok":        True,
                "mode":      "PAPER",
                "symbol":    symbol,
                "qty":       qty,
                "side":      side,
                "price":     price,
                "sl_price":  sl_price,
                "timestamp": datetime.now(IST).isoformat(),
                "order_id":  f"PAPER_{int(time.time())}",
            }
            log.info(f"[DRY-RUN] Order: {result}")
            return result

        if not self._conn:
            return {"ok": False, "error": "Not connected"}

        try:
            token   = self._get_token(symbol)
            exch    = "NSE"
            actual_order_type = "LIMIT" if order_type == "LIMIT" else "MARKET"

            order_params = {
                "variety":       "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken":   token,
                "transactiontype": side,
                "exchange":      exch,
                "ordertype":     actual_order_type,
                "producttype":   product,
                "duration":      "DAY",
                "price":         str(price) if order_type == "LIMIT" else "0",
                "quantity":      str(qty),
                "triggerprice":  "0",
            }
            resp = self._conn.placeOrder(order_params)

            # Place SL order
            sl_params = {
                **order_params,
                "variety":         "STOPLOSS",
                "transactiontype": "SELL" if side == "BUY" else "BUY",
                "ordertype":       "STOPLOSS_LIMIT",
                "price":           str(sl_price),
                "triggerprice":    str(sl_price * 0.999 if side == "BUY"
                                       else sl_price * 1.001),
            }
            sl_resp = self._conn.placeOrder(sl_params)

            log.info(
                f"[LIVE] Order placed: {side} {qty} {symbol} @ {price}  "
                f"SL={sl_price}  order_id={resp.get('data',{}).get('orderid')}"
            )
            return {
                "ok":         True,
                "mode":       "LIVE",
                "order_id":   resp.get("data", {}).get("orderid", ""),
                "sl_order_id": sl_resp.get("data", {}).get("orderid", ""),
                "symbol":     symbol,
                "qty":        qty,
                "side":       side,
                "price":      price,
            }

        except Exception as e:
            log.error(f"Order placement failed {symbol}: {e}")
            return {"ok": False, "error": str(e)}

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> dict:
        if not self.live or not self._conn:
            return {"ok": True, "mode": "PAPER", "order_id": order_id}
        try:
            resp = self._conn.cancelOrder(order_id, variety)
            return {"ok": True, "order_id": order_id, "resp": resp}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_positions(self) -> list[dict]:
        if not self.live or not self._conn:
            return []
        try:
            resp = self._conn.position()
            return resp.get("data", []) or []
        except Exception as e:
            log.warning(f"Positions fetch: {e}")
            return []

    def get_portfolio(self) -> list[dict]:
        if not self.live or not self._conn:
            return []
        try:
            resp = self._conn.holding()
            return resp.get("data", []) or []
        except Exception as e:
            log.warning(f"Portfolio fetch: {e}")
            return []

    def get_funds(self) -> dict:
        if not self.live or not self._conn:
            return {"available": 0, "used": 0}
        try:
            resp = self._conn.rmsLimit()
            d = resp.get("data", {})
            return {
                "available": float(d.get("availablecash", 0)),
                "used":      float(d.get("utiliseddebits", 0)),
                "total":     float(d.get("net", 0)),
            }
        except Exception as e:
            log.warning(f"Funds fetch: {e}")
            return {"available": 0, "used": 0}

    def close_all_positions(self) -> list[dict]:
        """Emergency exit: close all open positions at market price."""
        results = []
        for pos in self.get_positions():
            sym  = pos.get("tradingsymbol", "")
            qty  = abs(int(pos.get("netqty", 0)))
            side = "SELL" if int(pos.get("netqty", 0)) > 0 else "BUY"
            if qty > 0:
                r = self.place_order(sym, qty, side, 0, 0, order_type="MARKET")
                results.append(r)
                time.sleep(0.5)
        return results

    # ── Historical data ───────────────────────────────────────────────────────

    def get_historical(self, symbol: str, interval: str = "ONE_DAY",
                        from_date: str = "2024-01-01",
                        to_date: str | None = None) -> list[dict]:
        """
        Fetch historical candles from Angel One.
        interval: ONE_MINUTE, FIVE_MINUTE, ONE_HOUR, ONE_DAY
        """
        if not self.live or not self._conn:
            return []
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        try:
            token = self._get_token(symbol)
            params = {
                "exchange":    "NSE",
                "symboltoken": token,
                "interval":    interval,
                "fromdate":    from_date + " 09:00",
                "todate":      to_date,
            }
            resp = self._conn.getCandleData(params)
            return resp.get("data", []) or []
        except Exception as e:
            log.warning(f"Historical {symbol}: {e}")
            return []


# ─── Singleton ────────────────────────────────────────────────────────────────

_broker_instance: AngelOneBroker | None = None


def get_broker(live: bool = False) -> AngelOneBroker:
    global _broker_instance
    if _broker_instance is None:
        _broker_instance = AngelOneBroker(live=live)
    return _broker_instance
