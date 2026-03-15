#!/usr/bin/env bash
# logs.sh — View and tail agent cron logs
#
# Usage:
#   ./logs.sh              # show last 50 lines
#   ./logs.sh -f           # tail -f (live follow)
#   ./logs.sh -n 100       # show last 100 lines
#   ./logs.sh --errors     # show only ERROR/FATAL/PARTIAL lines
#   ./logs.sh --signals    # show only signal scan summaries
#   ./logs.sh --today      # show only today's entries

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$PROJECT_DIR/data/agents.log"

if [ ! -f "$LOG" ]; then
    echo "No log file yet: $LOG"
    echo "Logs appear after first cron run."
    exit 0
fi

case "${1:-}" in
    -f|--follow)
        echo "Following $LOG (Ctrl+C to stop)..."
        tail -f "$LOG"
        ;;
    --errors)
        echo "=== Errors / Warnings ==="
        grep -E "ERROR|FATAL|WARNING|PARTIAL|ERRORS=" "$LOG" | tail -50
        ;;
    --signals)
        echo "=== Signal scan results ==="
        grep "signals:" "$LOG" | tail -20
        ;;
    --today)
        today=$(date +%Y-%m-%d)
        echo "=== Today ($today) ==="
        grep "^$today" "$LOG"
        ;;
    -n)
        n="${2:-50}"
        tail -n "$n" "$LOG"
        ;;
    *)
        echo "=== Last 50 lines of $LOG ==="
        tail -50 "$LOG"
        ;;
esac
