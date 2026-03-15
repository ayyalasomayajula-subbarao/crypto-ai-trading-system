#!/usr/bin/env bash
# cron_setup.sh — Install automated agent schedules for crypto-ai-system
#
# Usage:
#   chmod +x cron_setup.sh
#   ./cron_setup.sh          # install all jobs
#   ./cron_setup.sh --remove # remove all jobs

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/venv/bin/python"
RUNNER="$PROJECT_DIR/agent_runner.py"
LOG="$PROJECT_DIR/data/agents.log"
MARKER="# crypto-ai-system"

# ── Verify python exists ──────────────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: venv not found at $PYTHON"
    echo "Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# ── Cron job definitions ──────────────────────────────────────────────────────
# Format: "schedule|task|description"
JOBS=(
    "15 */6 * * *|update-data|Data update every 6h (offset 15min from hour)"
    "30 */4 * * *|scan|Signal scan every 4h (30min after candle close)"
    "0  8  * * *|report|Daily risk report at 8am"
    "0  2  * * 0|self-improve|Weekly self-improve Sunday 2am"
)

# ── Build cron lines ──────────────────────────────────────────────────────────
build_cron_lines() {
    for job in "${JOBS[@]}"; do
        schedule="${job%%|*}"
        rest="${job#*|}"
        task="${rest%%|*}"
        desc="${rest##*|}"
        echo "$MARKER ($desc)"
        echo "$schedule cd \"$PROJECT_DIR\" && \"$PYTHON\" \"$RUNNER\" --cron $task >> \"$LOG\" 2>&1"
    done

    # Log rotation — weekly Sunday 3am
    echo "$MARKER (Weekly log rotation)"
    echo "0 3 * * 0 cd \"$PROJECT_DIR\" && mv -f \"$LOG\" \"${LOG%.log}.\$(date +\%Y\%m\%d).log\" 2>/dev/null; find \"$PROJECT_DIR/data\" -name 'agents.*.log' -mtime +30 -delete 2>/dev/null; true"
}

# ── Install ───────────────────────────────────────────────────────────────────
install_cron() {
    echo "Installing cron jobs for: $PROJECT_DIR"

    # Get current crontab (ignore error if empty)
    existing=$(crontab -l 2>/dev/null || true)

    # Remove any old crypto-ai-system entries
    cleaned=$(echo "$existing" | grep -v "$MARKER" | grep -v "agent_runner.py" || true)

    # Build new jobs
    new_jobs=$(build_cron_lines)

    # Write combined crontab
    {
        echo "$cleaned"
        echo ""
        echo "# ── Crypto AI Trading System ────────────────────────────────────"
        echo "$new_jobs"
        echo ""
    } | crontab -

    echo ""
    echo "Installed jobs:"
    crontab -l | grep -A1 "$MARKER" | grep -v "^--$"
    echo ""
    echo "Log file: $LOG"
    echo ""
    echo "To verify: crontab -l"
    echo "To remove: ./cron_setup.sh --remove"
}

# ── Remove ────────────────────────────────────────────────────────────────────
remove_cron() {
    echo "Removing crypto-ai-system cron jobs..."
    existing=$(crontab -l 2>/dev/null || true)
    cleaned=$(echo "$existing" \
        | grep -v "$MARKER" \
        | grep -v "agent_runner.py" \
        | grep -v "Crypto AI Trading System" \
        | grep -v "agents\.\*.log" \
        || true)
    echo "$cleaned" | crontab -
    echo "Done. Remaining crontab:"
    crontab -l || echo "(empty)"
}

# ── Main ──────────────────────────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/data"

if [ "${1:-}" = "--remove" ]; then
    remove_cron
else
    install_cron
fi
