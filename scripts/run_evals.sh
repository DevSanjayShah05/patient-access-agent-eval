#!/usr/bin/env bash
# All three agent configurations on all 100 scenarios (resumable). ~1.5-2.5 h each on an Apple M2 with qwen2.5:7b.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
.venv/bin/python -m front_desk.cli run
.venv/bin/python -m front_desk.cli run --guarded --guard-version 1
.venv/bin/python -m front_desk.cli run --guarded --guard-version 2
