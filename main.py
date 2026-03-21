"""
main.py — Single-port gateway combining Crypto AI + India Stocks APIs.

Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

All crypto routes:  /analyze/{coin}, /scan, /prices, /agents/*, /ws/prices, ...
All stocks routes:  /stocks/scan, /stocks/prices, /stocks/market/*, /ws/stocks/prices, ...

Module isolation strategy
─────────────────────────
Both apps use modules named 'precision_verdict' and 'paper_trader' but with
completely different implementations. The steps below ensure each app gets
its own version while preserving lazy-import correctness at runtime:

1. Import api_final  →  crypto versions land in sys.modules
2. Save crypto module references
3. Evict conflicting names from sys.modules
4. Add india-stocks/ to sys.path and import api_stocks  →  stocks versions load
5. Restore crypto versions in sys.modules so api_final's lazy imports work
"""

import sys
import os
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_INDIA_STOCKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "india-stocks")

# ── 1. Import crypto API (loads root/precision_verdict + root/paper_trader) ───
import api_final as _crypto_mod

# ── 2. Save references to crypto-specific modules before eviction ─────────────
_crypto_modules: dict = {}
for _name in ("precision_verdict", "paper_trader"):
    if _name in sys.modules:
        _crypto_modules[_name] = sys.modules[_name]

# ── 3. Evict so api_stocks can load its own fresh versions ────────────────────
for _name in _crypto_modules:
    sys.modules.pop(_name, None)

# ── 4. Add india-stocks path and import stocks API ────────────────────────────
if _INDIA_STOCKS not in sys.path:
    sys.path.insert(0, _INDIA_STOCKS)

import api_stocks as _stocks_mod   # loads india-stocks/precision_verdict + paper_trader

# ── 5. Restore crypto versions so api_final's lazy imports keep working ───────
sys.modules.update(_crypto_modules)

# ── Combined app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Trading System",
    version="1.0.0",
    description="Crypto AI + India Stocks — combined API on port 8000",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Merge routes ───────────────────────────────────────────────────────────────
# Crypto: /analyze, /scan, /prices, /agents, /ws/prices, ...
# Stocks: /stocks/*, /ws/stocks/prices
# Skip conflicting / and /health — redefined below.
# IMPORTANT: api_final has a catch-all /{full_path:path} SPA route — it must go
# LAST, after all stocks routes, otherwise it intercepts /stocks/scan etc.

_SKIP = {"/", "/health"}
_catchall_routes = []   # deferred: must be appended after stocks routes

for _route in _crypto_mod.app.routes:
    _path = getattr(_route, "path", None)
    if _path in _SKIP:
        continue
    # Defer catch-all path routes (e.g. /{full_path:path}) to after stocks routes
    if _path and _path.endswith(":path}"):
        _catchall_routes.append(_route)
        continue
    app.routes.append(_route)

for _route in _stocks_mod.app.routes:
    if getattr(_route, "path", None) not in _SKIP:
        app.routes.append(_route)

# Add catch-all routes last so /stocks/* API paths are matched first
for _route in _catchall_routes:
    app.routes.append(_route)


# ── Root + health ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "AI Trading System",
        "version": "1.0.0",
        "status":  "running",
        "port":    8000,
        "apis": {
            "crypto": ["/analyze/{coin}", "/scan", "/prices", "/ai-analysis/{coin}",
                       "/agents/*", "/paper-trading/*", "/ws/prices"],
            "stocks": ["/stocks/scan", "/stocks/prices", "/stocks/market/overview",
                       "/stocks/verdict/{symbol}", "/stocks/paper-trading/*",
                       "/stocks/backtest/*", "/ws/stocks/prices"],
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Startup — trigger both sub-apps' background tasks ────────────────────────

@app.on_event("startup")
async def startup():
    for handler in _crypto_mod.app.router.on_startup:
        await handler()
    for handler in _stocks_mod.app.router.on_startup:
        await handler()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
