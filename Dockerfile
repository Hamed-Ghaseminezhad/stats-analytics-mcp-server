FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
COPY scripts ./scripts
RUN .venv/bin/python scripts/make_demo_db.py /app/demo/demo.db

FROM python:3.12-slim
RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/demo /app/demo
ENV PATH="/app/.venv/bin:$PATH" STATS_DB_PATH=/app/demo/demo.db PYTHONUNBUFFERED=1
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8000), 2)"
# Listening on 0.0.0.0 inside the container. The server refuses to start without MCP_AUTH_TOKEN.
CMD ["stats-analytics-mcp-server", "--transport", "streamable-http", "--host", "0.0.0.0"]
