"""Agent Memory — SQLite, one connection per call (thread-safe)."""
from __future__ import annotations
import sqlite3
import json
import time
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import AGENT_DB_PATH

def _conn():
    os.makedirs(os.path.dirname(AGENT_DB_PATH), exist_ok=True)
    c = sqlite3.connect(AGENT_DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS agent_memory
                 (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)""")
    c.commit()
    return c

def store(key: str, value: dict) -> None:
    c = _conn()
    c.execute("INSERT OR REPLACE INTO agent_memory VALUES (?,?,?)",
              (key, json.dumps(value), time.time()))
    c.commit(); c.close()

def fetch(key: str) -> dict | None:
    c = _conn()
    row = c.execute("SELECT value FROM agent_memory WHERE key=?", (key,)).fetchone()
    c.close()
    return json.loads(row[0]) if row else None

def fetch_all(prefix: str = "") -> list[dict]:
    c = _conn()
    rows = c.execute(
        "SELECT key, value, updated_at FROM agent_memory WHERE key LIKE ?",
        (f"{prefix}%",)
    ).fetchall()
    c.close()
    return [{"key": r[0], "value": json.loads(r[1]), "ts": r[2]} for r in rows]
