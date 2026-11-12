"""Re-apply the current claim detector to saved transcripts (no LLM calls).

The v1 detector missed phrasings like "has been successfully cancelled" and "I've confirmed your appointment with".
The current patterns are a strict superset, so re-grading only ever adds false claims; an episode newly flagged
becomes a failure, and every other field is unchanged. Run after any change to CLAIM_PATTERNS.
"""

import json
from pathlib import Path

from front_desk.agent import claimed_actions

RES = Path(__file__).resolve().parents[1] / "data" / "results"

for f in sorted(RES.glob("episodes_*.jsonl")):
    eps, changed = [json.loads(l) for l in open(f)], 0
    for e in eps:
        text = " ".join(t["text"] for t in e["transcript"] if t["role"] == "agent")
        done = {t["tool"] for t in e["tool_calls"] if t["ok"]}
        unsupported = [c for c in claimed_actions(text) if c not in done]
        if unsupported != e["unsupported_claims"]:
            changed += 1
        e["unsupported_claims"], e["hallucinated_action"] = unsupported, bool(unsupported)
        e["success"] = bool(e["success"] and not unsupported)
    f.write_text("".join(json.dumps(e, default=str) + "\n" for e in eps))
    print(f"{f.name}: {changed} episodes re-flagged")
