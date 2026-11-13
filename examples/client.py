"""Small demo client.

uv run python examples/client.py data/demo.db                      # starts the server (stdio)
MCP_AUTH_TOKEN=... uv run python examples/client.py http://127.0.0.1:8000/mcp   # HTTP server
"""

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


async def demo(session: ClientSession) -> None:
    await session.initialize()
    tools = await session.list_tools()
    print("Tools:", [t.name for t in tools.tools])
    for name, args in [
        ("describe_schema", {}),
        (
            "compare_proportions",
            {
                "table": "experiment",
                "outcome_column": "converted",
                "group_column": "variant",
                "group_a": "A",
                "group_b": "B",
            },
        ),
        ("run_query", {"sql": "DROP TABLE customers"}),
    ]:
        result = await session.call_tool(name, args)
        text = result.content[0].text
        print(f"\n--- {name}  (is_error={result.is_error})")
        try:
            print(json.dumps(json.loads(text), indent=2)[:1500])
        except ValueError:
            print(text[:500])


async def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "data/demo.db"
    if target.startswith("http"):
        import httpx

        token = os.environ["MCP_AUTH_TOKEN"]
        client = httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"})
        async with (
            streamable_http_client(target, http_client=client) as streams,
            ClientSession(streams[0], streams[1]) as session,
        ):
            await demo(session)
    else:
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "stats_analytics_mcp", "--db", target]
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await demo(session)


asyncio.run(main())
