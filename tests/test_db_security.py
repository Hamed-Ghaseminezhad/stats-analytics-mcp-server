import hashlib

import pytest

from stats_analytics_mcp.db import QueryError, ReadOnlyDB


@pytest.fixture()
def db(demo_db):
    database = ReadOnlyDB(demo_db, timeout_s=0.5)
    yield database
    database.close()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


BAD_SQL = [
    "DROP TABLE customers",
    "DELETE FROM customers",
    "INSERT INTO customers (age) VALUES (1)",
    "UPDATE customers SET age = 0",
    "SELECT 1; DROP TABLE customers",
    "PRAGMA writable_schema = ON",
    "ATTACH DATABASE '/tmp/x.db' AS x",
    "CREATE TABLE t (a INT)",
    "WITH x AS (SELECT 1) DELETE FROM customers",
    "SELECT load_extension('/tmp/evil.so')",
    "SELECT * FROM sqlite_master",
    "SELECT readfile('/etc/passwd')",
    "",
    "   ;  ",
    "not sql at all (",
]


@pytest.mark.parametrize("sql", BAD_SQL)
def test_bad_sql_is_rejected(db, demo_db, sql):
    before = sha(demo_db)
    with pytest.raises(QueryError):
        db.run_select(sql)
    assert sha(demo_db) == before  # the file was never touched


def test_normal_select_works(db):
    out = db.run_select(
        "SELECT segment, COUNT(*) AS n FROM customers GROUP BY segment ORDER BY n DESC"
    )
    assert out["columns"] == ["segment", "n"]
    assert out["row_count"] == 3 and out["truncated"] is False


def test_cte_and_union_work(db):
    out = db.run_select("WITH a AS (SELECT 1 AS x) SELECT x FROM a UNION SELECT 2")
    assert out["row_count"] == 2


def test_row_cap_and_truncated_flag(db):
    out = db.run_select("SELECT * FROM customers", max_rows=10)
    assert out["row_count"] == 10 and out["truncated"] is True


def test_runaway_query_times_out(db):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c"
    with pytest.raises(QueryError, match="longer than"):
        db.run_select(sql)


def test_long_text_is_shortened_and_marked(db):
    out = db.run_select("SELECT text FROM support_tickets WHERE ticket_id = 3")
    cell = out["rows"][0][0]
    assert len(cell) < 260 and cell.endswith("...[truncated]")


def test_control_characters_removed(db):
    out = db.run_select("SELECT 'a' || char(0) || 'b' || char(27) || 'c'")
    assert out["rows"][0][0] == "a b c"


def test_giant_string_blocked_by_sqlite_limit(db):
    with pytest.raises(QueryError):
        db.run_select("SELECT replace(hex(zeroblob(10)), '0', hex(zeroblob(2000000)))")


def test_unknown_table_and_column_rejected(db):
    with pytest.raises(QueryError, match="Unknown table"):
        db.fetch_column("nope", "x")
    with pytest.raises(QueryError, match="Unknown column"):
        db.fetch_column("customers", 'age"; DROP TABLE customers; --')
