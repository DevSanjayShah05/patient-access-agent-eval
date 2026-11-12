"""python -m front_desk.cli run [--guarded [--guard-version 1|2]] [--limit N]   |   python -m front_desk.cli evaluate"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import FrontDeskAgent
from .clinic import build_clinic
from .runner import run_episode
from .scenarios import generate

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results"


SPLITS = {"dev": (11, "scenarios.json", RES), "test": (2027, "scenarios_test.json", RES / "test")}


def cmd_run(guarded: bool, model: str, limit: int | None, guard_version: int = 2, split: str = "dev") -> None:
    """dev = the 100 scenarios used while designing the guardrails; test = 100 held-out scenarios (seed 2027),
    fixed 2026-11-12 before any further tuning and run only for final numbers."""
    seed, scen_file, res = SPLITS[split]
    res.mkdir(parents=True, exist_ok=True)
    scenarios, base = generate(build_clinic(), n_per_type=10, seed=seed)
    (ROOT / "data" / scen_file).write_text(json.dumps(
        [{"scenario_id": s.scenario_id, "kind": s.kind, "opening": s.opening, "expect": s.expect,
          "persona": {k: v for k, v in s.persona.items()}} for s in scenarios], indent=1, default=str))
    agent = FrontDeskAgent(model=model, guarded=guarded, guard_version=guard_version)
    out = res / f"episodes_{agent.name.replace(':', '-').replace('+', '_')}.jsonl"
    done = {json.loads(l)["scenario_id"] for l in open(out)} if out.exists() else set()
    todo = [s for s in scenarios if s.scenario_id not in done][: limit or None]
    print(f"{agent.name}: {len(done)} done, {len(todo)} to go", flush=True)
    with open(out, "a") as f:
        for i, s in enumerate(todo, 1):
            r = run_episode(s, base, agent)
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            print(f"  {i}/{len(todo)} {s.scenario_id} success={r['success']} {r['wall_seconds']}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--guarded", action="store_true")
    r.add_argument("--model", default="qwen2.5:7b")
    r.add_argument("--limit", type=int)
    r.add_argument("--guard-version", type=int, default=2, choices=[1, 2, 3])
    r.add_argument("--split", choices=["dev", "test"], default="dev")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--split", choices=["dev", "test"], default="dev")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.guarded, a.model, a.limit, a.guard_version, a.split)
    else:
        from .evaluate import run_all
        run_all(a.split)


if __name__ == "__main__":
    main()
