"""Aggregate episode results into the metrics a patient-access team would track."""

from __future__ import annotations

import json
from math import comb, sqrt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results"
PRETTY = {"qwen2.5:7b": "Baseline agent", "qwen2.5:7b+guardrails": "Guardrails v1 (block only)",
          "qwen2.5:7b+guardrails_v2": "Guardrails v2"}
BASELINE = "qwen2.5:7b"


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(RES.glob("episodes_*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    return pd.DataFrame(rows)


def system_metrics(d: pd.DataFrame) -> dict:
    n = len(d)
    urgent = d[d.kind == "urgent_symptom"]
    non_urgent = d[d.kind != "urgent_symptom"]
    resolvable = d[~d.kind.isin(["urgent_symptom", "identity_mismatch", "unavailable_service"])]
    blocked = d.tool_calls.map(lambda tc: sum(t.get("blocked", False) for t in tc))
    k = int(d.success.sum())
    return {
        "episodes": n,
        "task_success": k / n, "task_success_ci95": wilson(k, n),
        "episodes_with_policy_violation": float((d.violations.map(len) > 0).mean()),
        "episodes_with_false_claim": float(d.hallucinated_action.mean()),
        "urgent_escalation_recall": float(urgent.escalation_reasons.map(lambda r: any("urgent" in x.lower() for x in r)).mean())
        if len(urgent) else None,
        "unnecessary_escalation_rate": float(non_urgent.escalated.mean()),
        "containment_resolvable": float(((resolvable.success) & (~resolvable.escalated)).mean()),
        "median_llm_calls": float(d.llm_calls.median()), "median_seconds": float(d.wall_seconds.median()),
        "guard_blocks_total": int(blocked.sum()), "episodes_with_guard_block": int((blocked > 0).sum()),
        "violation_types": pd.Series([v for vs in d.violations for v in vs]).value_counts().to_dict(),
        "false_claim_types": pd.Series([c for cs in d.unsupported_claims for c in cs]).value_counts().to_dict(),
        "errors": int(d.error.notna().sum()),
    }


def run_all() -> dict:
    df = load()
    out = {"systems": {}, "by_kind": {}, "pretty": PRETTY}
    for sysname, d in df.groupby("system"):
        out["systems"][sysname] = system_metrics(d)
        out["by_kind"][sysname] = d.groupby("kind").agg(
            success=("success", "mean"), violation=("violations", lambda s: (s.map(len) > 0).mean()),
            false_claim=("hallucinated_action", "mean"), n=("success", "size")).round(3).to_dict("index")
    piv = df.pivot_table(index="scenario_id", columns="system", values="success", aggfunc="first")
    out["paired"] = {}
    for other in [x for x in sorted(df.system.unique()) if x != BASELINE and BASELINE in piv]:
        pr = piv[[BASELINE, other]].dropna()
        only_o = int(((pr[other] == 1) & (pr[BASELINE] == 0)).sum())
        only_b = int(((pr[BASELINE] == 1) & (pr[other] == 0)).sum())
        out["paired"][other] = {"vs": BASELINE, "n_pairs": int(len(pr)), "only_this_succeeds": only_o,
                                "only_baseline_succeeds": only_b, "mcnemar_exact_p": mcnemar_exact(only_b, only_o)}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    rows = []
    for s, kinds in out["by_kind"].items():
        for k, v in kinds.items():
            rows.append({"system": PRETTY.get(s, s), "kind": k, **v})
    pd.DataFrame(rows).to_csv(RES / "by_kind.csv", index=False)
    for s, m in out["systems"].items():
        print(f"{PRETTY.get(s, s):22s} success {m['task_success']:.2f} {[round(x, 2) for x in m['task_success_ci95']]} "
              f"violations {m['episodes_with_policy_violation']:.2f} false-claims {m['episodes_with_false_claim']:.2f} "
              f"urgent recall {m['urgent_escalation_recall']} containment {m['containment_resolvable']:.2f} n={m['episodes']}")
    for k, v in out["paired"].items():
        print(PRETTY.get(k, k), v)
    return out
