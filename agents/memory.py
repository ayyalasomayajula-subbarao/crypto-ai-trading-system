"""
AgentMemory — thread-safe SQLite memory store for all agents.

Tables:
  memory         — general key/value store with task scoping
  model_versions — tracks trained model history per coin (Upgrade 4)

Thread safety: creates a new sqlite3.connect() per call.
"""

import os
import json
import time
import sqlite3
from typing import Any, Optional, List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "agent_memory.db")


class AgentMemory:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        self.prune_old_entries()  # enforce retention on every startup

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """New connection per call — thread-safe."""
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id     TEXT    NOT NULL DEFAULT 'global',
                    agent_name  TEXT    NOT NULL,
                    key         TEXT    NOT NULL,
                    value       TEXT    NOT NULL,
                    created_at  REAL    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_mem_key  ON memory(key);
                CREATE INDEX IF NOT EXISTS idx_mem_task ON memory(task_id);

                CREATE TABLE IF NOT EXISTS model_versions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin        TEXT    NOT NULL,
                    model_path  TEXT    NOT NULL,
                    sharpe      REAL,
                    wr          REAL,
                    threshold   REAL,
                    created_at  REAL    NOT NULL,
                    active      INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_mv_coin ON model_versions(coin);
            """)

    # ── General memory ────────────────────────────────────────────────────────

    def store(
        self,
        key: str,
        value: Any,
        agent_name: str = "system",
        task_id: str = "global",
    ):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory (task_id, agent_name, key, value, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_id, agent_name, key, json.dumps(value), time.time()),
            )

    def retrieve(self, key: str) -> Optional[Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM memory WHERE key = ? ORDER BY created_at DESC LIMIT 1",
                (key,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def get_task_context(self, task_id: str, limit: int = 10) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT agent_name, key, value, created_at "
                "FROM memory WHERE task_id = ? ORDER BY created_at DESC LIMIT ?",
                (task_id, limit),
            ).fetchall()
        return [
            {"agent": r[0], "key": r[1], "value": json.loads(r[2]), "ts": r[3]}
            for r in rows
        ]

    def search(self, keyword: str, limit: int = 5) -> List[Dict]:
        kw = f"%{keyword}%"
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT agent_name, key, value FROM memory "
                "WHERE key LIKE ? OR value LIKE ? ORDER BY created_at DESC LIMIT ?",
                (kw, kw, limit),
            ).fetchall()
        return [{"agent": r[0], "key": r[1], "value": json.loads(r[2])} for r in rows]

    def prune_old_entries(self, days: int = 30):
        """Delete memory entries older than `days` days. model_versions are kept forever."""
        cutoff = time.time() - days * 86400
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM memory WHERE created_at < ?", (cutoff,)
            )
            if cur.rowcount:
                pass  # silent — avoid log noise on startup

    def summarize_and_store(self, agent_name: str, output: Any, task_id: str):
        """Compress large output to key stats before storing (token efficiency)."""
        if isinstance(output, list) and len(output) > 20:
            summary = output[:5]  # keep first 5
        elif isinstance(output, dict) and len(json.dumps(output)) > 4000:
            # Keep only top-level keys with scalar values
            summary = {k: v for k, v in output.items() if not isinstance(v, (list, dict))}
        else:
            summary = output
        self.store(f"{agent_name}_output", summary, agent_name=agent_name, task_id=task_id)

    # ── Model version tracking (Upgrade 4) ───────────────────────────────────

    def save_model_version(
        self,
        coin: str,
        model_path: str,
        sharpe: float,
        wr: float,
        threshold: float,
    ):
        self.deactivate_old_models(coin)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO model_versions "
                "(coin, model_path, sharpe, wr, threshold, created_at, active) "
                "VALUES (?, ?, ?, ?, ?, ?, 1)",
                (coin, model_path, sharpe, wr, threshold, time.time()),
            )

    def get_active_model(self, coin: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT coin, model_path, sharpe, wr, threshold, created_at "
                "FROM model_versions WHERE coin = ? AND active = 1 "
                "ORDER BY created_at DESC LIMIT 1",
                (coin,),
            ).fetchone()
        if not row:
            return None
        return {
            "coin": row[0],
            "model_path": row[1],
            "sharpe": row[2],
            "wr": row[3],
            "threshold": row[4],
            "created_at": row[5],
        }

    def deactivate_old_models(self, coin: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE model_versions SET active = 0 WHERE coin = ?", (coin,)
            )

    def list_model_history(self, coin: str, limit: int = 10) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT coin, model_path, sharpe, wr, threshold, created_at, active "
                "FROM model_versions WHERE coin = ? ORDER BY created_at DESC LIMIT ?",
                (coin, limit),
            ).fetchall()
        return [
            {
                "coin": r[0],
                "model_path": r[1],
                "sharpe": r[2],
                "wr": r[3],
                "threshold": r[4],
                "created_at": r[5],
                "active": bool(r[6]),
            }
            for r in rows
        ]


# Module-level singleton
memory = AgentMemory()
