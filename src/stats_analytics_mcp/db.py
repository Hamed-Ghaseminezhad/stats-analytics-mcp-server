"""Read-only SQLite access with several independent layers of protection.

Layers, from outside in:
1. sqlglot parses the SQL. Only a single SELECT-style statement is accepted.
2. The database file is opened read-only (SQLite URI mode=ro).
3. A SQLite authorizer denies everything except reading ordinary tables.
4. A progress handler aborts queries that run longer than the time limit.
5. SQLite size limits stop huge strings and huge SQL text.
6. The row count is capped and every returned cell is cleaned and shortened.
"""

from __future__ import annotations

import functools
import math
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

MAX_SQL_CHARS = 5_000
MAX_ROWS_HARD = 1_000
MAX_CELL_CHARS = 200
MAX_STAT_ROWS = 500_000
QUERY_TIMEOUT_S = 5.0

_ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)
_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter,
    exp.Command, exp.Pragma, exp.Attach, exp.Detach, exp.Transaction,
    exp.Commit, exp.Rollback, exp.Merge, exp.Set,
)  # fmt: skip
_DENIED_FUNCTIONS = {
    "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
    "zeroblob", "randomblob", "sqlite_compileoption_get",
}  # fmt: skip
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class QueryError(Exception):
    """A problem with the user's request. The message is safe to show to the model."""


def clean_value(value: Any) -> Any:
    """Make one database value safe to return: no control characters, bounded length."""
    if isinstance(value, bytes):
        return f"<blob {len(value)} bytes>"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, str):
        text = _CONTROL_CHARS.sub(" ", value)
        if len(text) > MAX_CELL_CHARS:
            text = text[:MAX_CELL_CHARS] + "...[truncated]"
        return text
    return value


def validate_select(sql: str) -> str:
    """Return the SQL without a trailing semicolon, or raise QueryError."""
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise QueryError("The SQL is empty.")
    if len(text) > MAX_SQL_CHARS:
        raise QueryError(f"The SQL is longer than {MAX_SQL_CHARS} characters.")
    try:
        statements = sqlglot.parse(text, read="sqlite")
    except SqlglotError as exc:
        raise QueryError(f"The SQL could not be parsed: {str(exc)[:200]}") from exc
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise QueryError("Send exactly one SQL statement.")
    tree = statements[0]
    if not isinstance(tree, _ALLOWED_ROOTS):
        raise QueryError("Only SELECT queries are allowed.")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise QueryError("Only read-only SELECT queries are allowed.")
    return text


def _authorizer(action: int, arg1: str | None, arg2: str | None, *_: Any) -> int:
    if action == sqlite3.SQLITE_SELECT or action == sqlite3.SQLITE_RECURSIVE:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        if arg1 and arg1.lower().startswith("sqlite_"):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        if arg2 and arg2.lower() in _DENIED_FUNCTIONS:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _locked(method):
    """Run a ReadOnlyDB method while holding the instance lock (SQLite handles are shared)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class ReadOnlyDB:
    def __init__(self, path: str | Path, timeout_s: float = QUERY_TIMEOUT_S) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Database file not found: {self.path}")
        self.timeout_s = timeout_s
        self._lock = threading.RLock()
        uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
        # Connection for user SQL: authorizer, limits and a time limit.
        self._query = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._query.execute("PRAGMA query_only = ON")
        self._query.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
        self._query.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_SQL_CHARS * 2)
        self._query.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50)
        self._query.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 10)
        self._query.set_authorizer(_authorizer)
        # Connection for our own fixed statements (schema, statistics). Identifiers are
        # checked against the real schema before they are ever placed in SQL.
        self._internal = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._internal.execute("PRAGMA query_only = ON")
        self._schema_cache: dict[str, list[tuple[str, str]]] | None = None

    def close(self) -> None:
        self._query.close()
        self._internal.close()

    # ---- schema -------------------------------------------------------------------
    @_locked
    def schema(self) -> dict[str, list[tuple[str, str]]]:
        if self._schema_cache is None:
            names = [
                r[0]
                for r in self._internal.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            cache: dict[str, list[tuple[str, str]]] = {}
            for name in names:
                info = self._internal.execute(f"PRAGMA table_info({quote_ident(name)})")
                cache[name] = [(row[1], row[2] or "") for row in info]
            self._schema_cache = cache
        return self._schema_cache

    @_locked
    def describe(self) -> list[dict[str, Any]]:
        out = []
        for table, columns in self.schema().items():
            count = self._internal.execute(f"SELECT COUNT(*) FROM {quote_ident(table)}").fetchone()[
                0
            ]
            out.append(
                {
                    "table": table,
                    "rows": count,
                    "columns": [{"name": c, "type": t} for c, t in columns],
                }
            )
        return out

    def check_table_column(self, table: str, *columns: str) -> None:
        schema = self.schema()
        if table not in schema:
            raise QueryError(f"Unknown table '{clean_value(table)}'. Call describe_schema first.")
        known = {c for c, _ in schema[table]}
        for column in columns:
            if column not in known:
                raise QueryError(
                    f"Unknown column '{clean_value(column)}' in table '{table}'. "
                    f"Columns: {sorted(known)}"
                )

    # ---- user SQL -----------------------------------------------------------------
    @_locked
    def run_select(self, sql: str, max_rows: int = 100) -> dict[str, Any]:
        text = validate_select(sql)
        max_rows = max(1, min(int(max_rows), MAX_ROWS_HARD))
        deadline = time.monotonic() + self.timeout_s
        self._query.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            cursor = self._query.execute(f"SELECT * FROM ({text}) LIMIT ?", (max_rows + 1,))
            columns = [d[0] for d in cursor.description or []]
            fetched = cursor.fetchall()
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise QueryError(
                    f"The query took longer than {self.timeout_s:.0f} seconds."
                ) from exc
            raise QueryError(f"SQLite rejected the query: {str(exc)[:200]}") from exc
        except sqlite3.DatabaseError as exc:
            raise QueryError(f"The query was not allowed: {str(exc)[:200]}") from exc
        finally:
            self._query.set_progress_handler(None, 0)
        truncated = len(fetched) > max_rows
        rows = [[clean_value(v) for v in row] for row in fetched[:max_rows]]
        return {
            "columns": [clean_value(c) for c in columns],
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
        }

    # ---- helpers for statistics ---------------------------------------------------
    @_locked
    def fetch_column(self, table: str, column: str) -> tuple[list[Any], bool]:
        self.check_table_column(table, column)
        sql = f"SELECT {quote_ident(column)} FROM {quote_ident(table)} LIMIT ?"
        rows = self._internal.execute(sql, (MAX_STAT_ROWS + 1,)).fetchall()
        capped = len(rows) > MAX_STAT_ROWS
        return [r[0] for r in rows[:MAX_STAT_ROWS]], capped

    @_locked
    def fetch_by_group(
        self, table: str, value_column: str, group_column: str, groups: list[str]
    ) -> tuple[dict[str, list[Any]], bool]:
        self.check_table_column(table, value_column, group_column)
        marks = ",".join("?" for _ in groups)
        sql = (
            f"SELECT CAST({quote_ident(group_column)} AS TEXT), {quote_ident(value_column)} "
            f"FROM {quote_ident(table)} WHERE CAST({quote_ident(group_column)} AS TEXT) IN ({marks}) "
            "LIMIT ?"
        )  # fmt: skip
        rows = self._internal.execute(sql, (*groups, MAX_STAT_ROWS + 1)).fetchall()
        capped = len(rows) > MAX_STAT_ROWS
        out: dict[str, list[Any]] = {g: [] for g in groups}
        for g, v in rows[:MAX_STAT_ROWS]:
            out[g].append(v)
        return out, capped

    @_locked
    def fetch_pairs(self, table: str, x: str, y: str) -> tuple[list[tuple[Any, Any]], bool]:
        self.check_table_column(table, x, y)
        sql = f"SELECT {quote_ident(x)}, {quote_ident(y)} FROM {quote_ident(table)} LIMIT ?"
        rows = self._internal.execute(sql, (MAX_STAT_ROWS + 1,)).fetchall()
        capped = len(rows) > MAX_STAT_ROWS
        return [(a, b) for a, b in rows[:MAX_STAT_ROWS]], capped
