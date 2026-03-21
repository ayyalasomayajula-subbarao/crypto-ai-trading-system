#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_stocks_to_ec2.sh
# Run this LOCALLY on your Mac to push the india-stocks system to EC2.
#
# Usage:
#   chmod +x deploy_stocks_to_ec2.sh
#   ./deploy_stocks_to_ec2.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── EDIT THESE ────────────────────────────────────────────────────────────────
EC2_USER="ubuntu"                          # change to "ec2-user" if Amazon Linux AMI
EC2_HOST="13.51.159.80"
EC2_KEY="~/Desktop/TradeWise.pem"
REMOTE_DIR="~/crypto-ai-system"           # where the repo lives on EC2
# ─────────────────────────────────────────────────────────────────────────────

SSH="ssh -i $EC2_KEY -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST"
SCP="scp -i $EC2_KEY -o StrictHostKeyChecking=no"

echo "=== [1/4] Pushing india-stocks/ code (excluding data CSVs) ==="
rsync -avz --progress \
  -e "ssh -i $EC2_KEY -o StrictHostKeyChecking=no" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='data/*.csv' \
  --exclude='data/*.json' \
  india-stocks/ \
  $EC2_USER@$EC2_HOST:$REMOTE_DIR/india-stocks/

echo ""
echo "=== [2/4] Pushing model files (gitignored, must scp separately) ==="
# Only push the 5 viable/marginal models — saves bandwidth
for SYMBOL in NIFTY50 BANKNIFTY NIFTYIT TITAN AXISBANK; do
  echo "  -> Pushing $SYMBOL model..."
  $SSH "mkdir -p $REMOTE_DIR/india-stocks/models/$SYMBOL"
  $SCP india-stocks/models/$SYMBOL/wf_decision_model_v2.pkl \
    $EC2_USER@$EC2_HOST:$REMOTE_DIR/india-stocks/models/$SYMBOL/
  $SCP india-stocks/models/$SYMBOL/decision_features_v2.txt \
    $EC2_USER@$EC2_HOST:$REMOTE_DIR/india-stocks/models/$SYMBOL/
  # wf_results.json (for backtest endpoint)
  if [ -f india-stocks/models/$SYMBOL/wf_results.json ]; then
    $SCP india-stocks/models/$SYMBOL/wf_results.json \
      $EC2_USER@$EC2_HOST:$REMOTE_DIR/india-stocks/models/$SYMBOL/
  fi
done

echo ""
echo "=== [3/4] Running server setup script on EC2 ==="
$SSH "bash $REMOTE_DIR/india-stocks/ec2_setup.sh"

echo ""
echo "=== [4/4] Done! ==="
echo "  Stocks API: http://$EC2_HOST:8001/health"
echo "  Scan all:   http://$EC2_HOST:8001/stocks/scan"
echo ""
echo "  To tail logs:  ssh -i $EC2_KEY $EC2_USER@$EC2_HOST 'tail -f /tmp/stocks_api.log'"
