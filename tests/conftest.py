import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def demo_db(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("db") / "demo.db"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_demo_db.py"), str(path)], check=True
    )
    return path
