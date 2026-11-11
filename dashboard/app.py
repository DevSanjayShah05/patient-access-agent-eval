"""Front Desk dashboard. Run: streamlit run dashboard/app.py"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "results"
COLORS = {"qwen2.5:7b": "#eb6834", "qwen2.5:7b+guardrails": "#2a78d6"}
INK2, GRID = "#52514e", "#e7e6e1"
KIND_LABEL = {"book_existing": "Book (existing patient)", "book_new_patient": "Book (new patient)",
              "reschedule": "Reschedule", "cancel": "Cancel (>24h)", "late_cancel": "Late cancel (<24h, $50 fee)",
              "insurance_not_accepted": "Plan not accepted", "urgent_symptom": "Urgent symptom",
              "identity_mismatch": "Identity fails", "info_request": "Hours / location", "unavailable_service": "Service not offered"}

st.set_page_config(page_title="Front Desk", layout="wide")


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=40, b=8), plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
                      font=dict(color=INK2, size=13), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


@st.cache_data
def load():
    s = json.loads((RES / "summary.json").read_text())
    eps = pd.DataFrame([json.loads(l) for f in sorted(RES.glob("episodes_*.jsonl")) for l in open(f)])
    return s, eps


S, E = load()
pretty = S["pretty"]
systems = [x for x in ["qwen2.5:7b", "qwen2.5:7b+guardrails"] if x in S["systems"]]

st.title("Front Desk")
st.caption("Evaluating an AI patient-access agent (scheduling, intake, triage) on 100 **SYNTHETIC** scenarios against a "
           "simulated clinic, graded on the clinic's final state, not on how the reply sounds.")
tabs = st.tabs(["Scorecard", "By scenario type", "Failure modes", "Conversation explorer"])

with tabs[0]:
    rows = []
    for s in systems:
        m = S["systems"][s]
        rows.append({"Configuration": pretty[s], "Task success": m["task_success"],
                     "95% CI": f"{m['task_success_ci95'][0]:.0%}-{m['task_success_ci95'][1]:.0%}",
                     "Policy violation": m["episodes_with_policy_violation"],
                     "Claimed an action that didn't happen": m["episodes_with_false_claim"],
                     "Urgent cases escalated": m["urgent_escalation_recall"],
                     "Contained (resolvable requests)": m["containment_resolvable"],
                     "Median LLM calls": m["median_llm_calls"], "Median seconds": m["median_seconds"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in
                                ["Task success", "Policy violation", "Claimed an action that didn't happen",
                                 "Urgent cases escalated", "Contained (resolvable requests)"]})
    if "paired" in S:
        p = S["paired"]
        st.markdown(f"Same {p['n_pairs']} scenarios for both: guardrails succeeded where the baseline failed in "
                    f"**{p['guardrails_only_success']}**, the reverse in **{p['baseline_only_success']}** "
                    f"(exact McNemar p = {p['mcnemar_exact_p']:.3g}).")

with tabs[1]:
    kinds = list(KIND_LABEL)
    fig = go.Figure()
    for s in systems:
        bk = S["by_kind"][s]
        fig.add_bar(y=[KIND_LABEL[k] for k in kinds], x=[bk.get(k, {}).get("success", 0) for k in kinds],
                    orientation="h", name=pretty[s], marker_color=COLORS[s],
                    hovertemplate="%{y}: %{x:.0%}<extra>" + pretty[s] + "</extra>")
    fig.update_layout(barmode="group", title="Task success by scenario type (10 scenarios each)",
                      xaxis_tickformat=".0%", xaxis_range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, 560), width="stretch")

with tabs[2]:
    c1, c2 = st.columns(2)
    for col, key, title in [(c1, "violation_types", "Policy violations"), (c2, "false_claim_types", "False claims (tool never succeeded)")]:
        with col:
            st.subheader(title)
            data = [{"configuration": pretty[s], "type": k, "episodes": v} for s in systems
                    for k, v in S["systems"][s][key].items()]
            st.dataframe(pd.DataFrame(data) if data else pd.DataFrame({"none": []}), hide_index=True, width="stretch")
    st.markdown("Guardrail blocks: " + ", ".join(
        f"{pretty[s]}: {S['systems'][s]['guard_blocks_total']} blocked tool calls" for s in systems))

with tabs[3]:
    sid = st.selectbox("Scenario", sorted(E.scenario_id.unique()))
    cols = st.columns(len(systems))
    for col, s in zip(cols, systems):
        r = E[(E.scenario_id == sid) & (E.system == s)]
        with col:
            if r.empty:
                st.info("not run")
                continue
            r = r.iloc[0]
            st.markdown(f"**{pretty[s]}**: {'✅ success' if r.success else '❌ failure'}")
            if r.violations:
                st.error("; ".join(r.violations))
            if r.unsupported_claims:
                st.warning("Claimed without a successful tool call: " + ", ".join(r.unsupported_claims))
            for t in r.transcript:
                st.markdown(f"**{'Patient' if t['role'] == 'patient' else 'Agent'}:** {t['text']}")
            st.caption("Tools: " + " → ".join(f"{x['tool']}{' (blocked)' if x.get('blocked') else '' if x['ok'] else ' (error)'}"
                                              for x in r.tool_calls))
