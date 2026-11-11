#!/usr/bin/env bash
# Both agent configurations on all 100 scenarios (resumable). ~2 h on an Apple M2 with qwen2.5:7b.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
.venv/bin/python -m front_desk.cli run
.venv/bin/python -m front_desk.cli run --guarded
