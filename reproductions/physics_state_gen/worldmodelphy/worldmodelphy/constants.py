from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DATA_ROOT = ARTIFACTS_ROOT / "data"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"
REPORT_ROOT = PROJECT_ROOT / "docs"
