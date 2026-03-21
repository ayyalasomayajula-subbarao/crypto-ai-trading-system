#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# ec2_setup.sh — Run on EC2 (called by deploy_stocks_to_ec2.sh)
# Sets up the india-stocks API service, cron jobs, and collects initial data.
# ─────────────────────────────────────────────────────────────────────────────

set -e

STOCKS_DIR="$HOME/crypto-ai-system/india-stocks"
LOG_DIR="/tmp"
PYTHON=$(which python3 || which python)

echo "=== [1/5] Installing Python dependencies ==="
pip3 install --quiet --upgrade \
  yfinance \
  pyotp \
  groq \
  openai \
  pytz \
  lightgbm \
  scikit-learn \
  fastapi \
  "uvicorn[standard]" \
  pandas \
  numpy \
  joblib \
  python-dotenv \
  requests

echo ""
echo "=== [2/5] Creating data directories ==="
mkdir -p $STOCKS_DIR/data
mkdir -p $STOCKS_DIR/data/option_chain

echo ""
echo "=== [3/5] Collecting initial NSE data (this may take 2-3 mins) ==="
cd $STOCKS_DIR
$PYTHON collect_nse_data.py && echo "  NSE data collected OK" || echo "  WARNING: NSE data collection failed — check yfinance connectivity"

echo ""
echo "=== [4/5] Setting up cron jobs ==="
# Remove any existing stocks cron entries to avoid duplicates
crontab -l 2>/dev/null | grep -v "stocks" | grep -v "india-stocks" | grep -v "collect_nse" > /tmp/crontab_clean.txt || true

# ── Daily data collection: Mon-Fri at 4:30pm IST = 11:00 UTC ──────────────
# ── Weekly data (Sunday midnight IST) + NSE holiday refresh ──────────────
cat >> /tmp/crontab_clean.txt << 'CRONEOF'

# ── India Stocks: daily data collection (4:30pm IST = 11:00 UTC, Mon-Fri) ──
0 11 * * 1-5 cd ~/crypto-ai-system/india-stocks && ACTIVE_SYMBOLS="NIFTY50,BANKNIFTY,NIFTYIT,TITAN,AXISBANK" python3 collect_nse_data.py >> /tmp/nse_collect.log 2>&1

# ── India Stocks: FII/DII data (5:00pm IST = 11:30 UTC, Mon-Fri) ──────────
30 11 * * 1-5 cd ~/crypto-ai-system/india-stocks && python3 collect_fii_dii.py >> /tmp/fii_dii.log 2>&1

# ── India Stocks: feature engineering after data (5:15pm IST = 11:45 UTC) ──
45 11 * * 1-5 cd ~/crypto-ai-system/india-stocks && ACTIVE_SYMBOLS="NIFTY50,BANKNIFTY,NIFTYIT,TITAN,AXISBANK" python3 feature_engineering.py >> /tmp/feature_eng.log 2>&1

# ── India Stocks: option chain snapshot (9:30am + 3:30pm IST = 04:00 + 10:00 UTC) ──
0 4 * * 1-5 cd ~/crypto-ai-system/india-stocks && python3 collect_option_chain.py --symbol NIFTY50 >> /tmp/option_chain.log 2>&1 && python3 collect_option_chain.py --symbol BANKNIFTY >> /tmp/option_chain.log 2>&1
0 10 * * 1-5 cd ~/crypto-ai-system/india-stocks && python3 collect_option_chain.py --symbol NIFTY50 >> /tmp/option_chain.log 2>&1 && python3 collect_option_chain.py --symbol BANKNIFTY >> /tmp/option_chain.log 2>&1

# ── India Stocks: auto-restart API if it crashes (every 5 min) ────────────
*/5 * * * * pgrep -f "api_stocks" > /dev/null || (cd ~/crypto-ai-system/india-stocks && nohup env ACTIVE_SYMBOLS="NIFTY50,BANKNIFTY,NIFTYIT,TITAN,AXISBANK" uvicorn api_stocks:app --host 0.0.0.0 --port 8001 --workers 1 >> /tmp/stocks_api.log 2>&1 &)

CRONEOF

crontab /tmp/crontab_clean.txt
echo "  Cron jobs installed:"
crontab -l | grep -E "stocks|nse|fii|option|feature" | sed 's/^/    /'

echo ""
echo "=== [5/5] Starting stocks API ==="
cd $STOCKS_DIR

# Kill any existing instance cleanly
pkill -f "api_stocks" 2>/dev/null && sleep 2 || true

# Production: restrict to viable/marginal symbols only to save RAM
export ACTIVE_SYMBOLS="NIFTY50,BANKNIFTY,NIFTYIT,TITAN,AXISBANK"

nohup env ACTIVE_SYMBOLS="$ACTIVE_SYMBOLS" uvicorn api_stocks:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 1 \
  >> $LOG_DIR/stocks_api.log 2>&1 &

sleep 3

# Health check
if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
  echo "  Stocks API is UP on port 8001 ✓"
  curl -s http://localhost:8001/health | python3 -m json.tool 2>/dev/null || true
else
  echo "  WARNING: Health check failed — check logs:"
  tail -20 $LOG_DIR/stocks_api.log
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Stocks API:   http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo 'YOUR_EC2_IP'):8001"
echo "  Health:       /health"
echo "  Scan:         /stocks/scan"
echo "  Paper status: /stocks/paper-trading/status"
echo "  Logs:         tail -f /tmp/stocks_api.log"
echo "═══════════════════════════════════════════════════════════"
