import pytest
from mcp.server.mcpserver.exceptions import ToolError
from starlette.testclient import TestClient

from stats_analytics_mcp import server

TOKEN = "x" * 32


@pytest.fixture(scope="module", autouse=True)
def configured(demo_db):
    server.configure(str(demo_db))


async def test_describe_schema_lists_tables():
    out = await server.describe_schema()
    assert {t["table"] for t in out["tables"]} == {"customers", "experiment", "support_tickets"}


async def test_run_query_carries_untrusted_note_and_blocks_writes():
    out = await server.run_query("SELECT text FROM support_tickets WHERE ticket_id = 3")
    assert "untrusted" in out["note"]
    with pytest.raises(ToolError):
        await server.run_query("DROP TABLE customers")


async def test_ab_test_conversion_difference_found():
    out = await server.compare_proportions("experiment", "converted", "variant", "A", "B")
    assert out["rate_b"] > out["rate_a"]
    assert out["n_a"] == 1500 and out["n_b"] == 1500
    assert out["warnings"]


async def test_compare_groups_on_spend_by_segment():
    out = await server.compare_groups("customers", "monthly_spend", "segment", "pro", "basic")
    assert out["mean_difference_a_minus_b"] > 40
    assert out["welch_p"] < 1e-6


async def test_correlation_tool():
    out = await server.correlation("customers", "tenure_months", "churned")
    assert out["pearson_r"] < 0  # longer tenure, less churn in the demo data


async def test_profile_tool():
    out = await server.profile_column("customers", "segment")
    assert out["kind"] == "categorical" and out["distinct"] == 3


async def test_errors_are_tool_errors():
    with pytest.raises(ToolError, match="Unknown table"):
        await server.profile_column("nope", "x")
    with pytest.raises(ToolError, match="different"):
        await server.compare_groups("customers", "age", "segment", "pro", "pro")
    with pytest.raises(ToolError, match="only 0 and 1"):
        await server.compare_proportions("customers", "age", "segment", "pro", "basic")


# ---- HTTP authentication -------------------------------------------------------------
INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
}  # fmt: skip
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


@pytest.fixture(scope="module")
def client():
    app = server.build_app(TOKEN)
    with TestClient(app, base_url="http://127.0.0.1:8000") as c:
        yield c


def test_http_without_token_is_rejected(client):
    assert client.post("/mcp", json=INIT, headers=HEADERS).status_code == 401


def test_http_with_wrong_token_is_rejected(client):
    r = client.post("/mcp", json=INIT, headers={**HEADERS, "Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_http_with_right_token_works(client):
    r = client.post("/mcp", json=INIT, headers={**HEADERS, "Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200 and "stats-analytics" in r.text


def test_oversized_body_rejected(client):
    big = {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {"pad": "x" * 200_000}}
    r = client.post("/mcp", json=big, headers={**HEADERS, "Authorization": f"Bearer {TOKEN}"})
    assert r.status_code in (400, 413)
