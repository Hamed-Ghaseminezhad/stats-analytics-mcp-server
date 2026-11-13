# Stats Analytics MCP Server

A read-only analytics server for AI assistants. It lets an assistant explore a SQL database and run statistical comparisons that come back with effect sizes, confidence intervals and plain warnings, not just a p-value.

Built with Python and the official MCP SDK (2.x).

## Why this exists

Most "chat with your database" tools only write SQL. When someone asks "did variant B work better?", they get a p-value and little else. This server makes the careful version the default: every statistical tool returns the size of the effect, an interval around it, and warnings about small samples, skew and causal claims.

## Example

On the included demo data, an A/B test with 1,500 customers per group:

| | Variant A | Variant B |
|---|---|---|
| Conversion | 11.5% | 13.7% |

The compare_proportions tool reports a rate difference (A minus B) of -2.3 points with a 95% interval of -4.6 to +0.1 points, Fisher’s exact p = 0.069, and a reminder to check that assignment was random. The interval includes zero, so the data does not settle the question. The tool shows that instead of declaring a winner.

## Tools

| Tool | What it does |
|---|---|
| `describe_schema` | Tables, columns and row counts |
| `run_query` | One read-only SELECT, with a row limit and a time limit |
| `profile_column` | Missing values, quantiles or top values for one column |
| `compare_groups` | Numeric column across two groups: Welch t-test, Mann-Whitney, mean difference with 95% CI, Hedges' g, Cliff's delta |
| `compare_proportions` | 0/1 outcome across two groups: rate difference (Newcombe CI), odds ratio, relative risk, Fisher's exact test |
| `correlation` | Pearson (with 95% CI) and Spearman |
| `adjust_p_values` | Holm, Benjamini-Hochberg or Bonferroni correction |

There is also a prompt, `rigorous_comparison`, that walks an assistant through a careful comparison.

## Security design

The server has no LLM inside and never writes. Several independent layers protect the database, and the tests check that each layer holds when the others are skipped.

| Risk | What the server does |
|---|---|
| Writes or schema changes | The file is opened read-only, `query_only` is on, and a SQLite authorizer denies everything except reading ordinary tables |
| SQL injection through tool arguments | Table and column names are checked against the real schema before use, and values are bound parameters |
| Multiple statements, `PRAGMA`, `ATTACH`, `load_extension` | Rejected by the SQL parser (sqlglot) and by the authorizer |
| Runaway queries | 5 second limit, row cap of 1,000, SQLite limits on string length and SQL length |
| Prompt injection hidden in data | Text cells are cleaned of control characters and cut to 200 characters. Every result carries a note that the values are untrusted data |
| Open HTTP server | Over HTTP the server needs a bearer token, compares it in constant time, refuses to listen on a non-loopback address without one, and caps request size at 64 KB |
| Container | Non-root user, no write access needed, port published to localhost only |

What this does not do: no filter can fully stop a model from being influenced by text inside data. The note and the shortened cells reduce the risk and do not remove it. Do not point this at a database that holds secrets.

## Quick start

You need [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Hamed-Ghaseminezhad/stats-analytics-mcp-server
cd stats-analytics-mcp-server
uv sync
uv run python scripts/make_demo_db.py data/demo.db
uv run python examples/client.py data/demo.db
uv run pytest
```

### Claude Desktop

```json
{
  "mcpServers": {
    "stats-analytics": {
      "command": "uv",
      "args": ["--directory", "/ABSOLUTE/PATH/stats-analytics-mcp-server", "run",
               "stats-analytics-mcp-server", "--db", "/ABSOLUTE/PATH/data/demo.db"]
    }
  }
}
```

### HTTP server

```bash
export MCP_AUTH_TOKEN="$(openssl rand -hex 24)"   # at least 24 characters
uv run stats-analytics-mcp-server --db data/demo.db --transport streamable-http
# listening on http://127.0.0.1:8000/mcp
```

Test it with the demo client: `uv run python examples/client.py http://127.0.0.1:8000/mcp`

### Docker

```bash
docker build -t stats-analytics-mcp .
export MCP_AUTH_TOKEN="$(openssl rand -hex 24)"
docker run --rm -p 127.0.0.1:8000:8000 -e MCP_AUTH_TOKEN \
  --read-only --cap-drop ALL --security-opt no-new-privileges stats-analytics-mcp
```

To use your own SQLite file, mount it read-only:
`-v "$(pwd)/mydata.db:/data/my.db:ro" -e STATS_DB_PATH=/data/my.db`

### Postman

1. New, then MCP, then HTTP.
2. URL: `http://localhost:8000/mcp`
3. Authorization tab: type Bearer Token, paste your `MCP_AUTH_TOKEN`.
4. Connect, pick a tool, fill the arguments, Run.

## Limits

- SQLite only, one database per server.
- Statistics assume independent observations. Repeated measures per customer need a different method.
- Group tests compare two groups at a time.

## Development

```bash
uv run ruff check src tests scripts
uv run pytest
```
