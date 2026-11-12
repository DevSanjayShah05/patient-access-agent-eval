# Front Desk

**Can an LLM agent run a clinic's front desk without lying to patients? An evaluation harness that grades a
scheduling agent on what it *did* to the clinic, not on how its replies sound.**

> A simulated multi-specialty practice, 100 **SYNTHETIC** patient conversations across 10 request types, a
> rule-based patient simulator and a state-based grader. Three versions of a local-LLM agent (qwen2.5:7b): no
> guardrails, a first guardrail design that backfired, and a second design built from that failure analysis.

![Conversation explorer: the same booking request handled by all three agent versions](docs/screenshots/04_explorer.png)

---

## The problem

Patient-access AI (scheduling, intake, triage by phone or chat) is one of the busiest areas in healthcare AI.
The demo is easy. The hard part is the failure that doesn't show up in a demo:

- The agent tells a patient *"you're booked for Wednesday at 8"*, but no booking exists.
- It books before checking insurance, or reveals someone's appointment to an unverified caller.
- It books a routine visit for someone describing chest pain.

A transcript that *reads* well can still contain any of these. So this harness grades each conversation on the
**clinic's final state** (appointments, registrations, fees, escalations) and on **policy compliance**, and checks
every claim the agent makes against the tool calls that actually succeeded.

## Results

All three agents use the same model, prompt, tools, scenarios and grader. The only difference is the guardrail
layer. Numbers come from `data/results/summary.json`.

| | Baseline | Guardrails v1 (block only) | **Guardrails v2** |
|---|---|---|---|
| Task success (95% CI) | 46% (37–56) | 51% (41–61) | **59% (49–68)** |
| Told the patient something happened that didn't | 40% | 38% | **0% detected**¹ |
| Urgent symptoms sent to a human + told to call 911/988 | 2 / 10 | 10 / 10 | **10 / 10** |
| Policy violations (graded outcomes) | 6% | 0% | 8% |
| Insured bookings made **without checking insurance** | **18 of 19** | 0 of 0 | **0 of 24** |
| Resolvable requests fully handled, no human needed | 39% | 31% | 41% |
| Median LLM calls per conversation | 9 | 9 | 9.5 |

**Paired comparison on the same 100 scenarios** (exact McNemar):
- v2 vs baseline: v2 succeeded where the baseline failed **15** times; the reverse happened **2** times (**p = 0.002**).
- v1 vs baseline: 12 vs 7 (p = 0.36), not a significant difference.
- v2 vs v1: 11 vs 3 (p = 0.057).

¹ v2's output validator uses the same phrase detector as the grader, so "0%" is partly built in. Reading v2's
transcripts by hand turned up **at least 2** false claims the detector misses ("I've reserved a spot for you…", "Dr. Lin
is confirmed"). Both conversations were already graded as failures. The honest v2 figure is "0% detected, ≥2% actual."

![Task success by scenario type](docs/screenshots/02_by_type.png)

## What happened: three versions, two lessons

**Baseline: policy in the prompt only.**
- It sounds competent and books readily.
- In **40%** of conversations it claimed an action that never happened, most often a booking or cancellation.
- It sent only **2 of 10** urgent cases (chest pain, stroke signs, suicidal thoughts) to a human.
- It skipped the required insurance check in **18 of 19** insured bookings. Task success only counts outcomes, so this
  cost nothing in the score. It is reported as a separate process metric.

**Guardrails v1: hard blocks.** Three deterministic layers:
1. An urgent-symptom safety net that bypasses the LLM.
2. A commit guard that blocks policy-violating tool calls and tells the model why.
3. An output validator that asks for one retry when a reply claims an action no tool performed.

Urgent recall went to 10/10 and graded violations to zero. **But booking success fell to 0 of 20.** The guard correctly
blocked bookings without an insurance check. The 7B model then checked insurance and **never retried the booking**,
yet told the patient *"I've confirmed your appointment with Dr. Lin."* The detector also missed that phrasing (see
below). **Lesson 1: a block that relies on the model to recover is a new way to fail, not a safety net.**

**Guardrails v2: enforce preconditions in code.**
- The commit guard runs the insurance check itself, then books or blocks. It no longer hopes the model retries.
- If the validator's retry still claims an action that never happened, the patient gets a safe question
  ("I haven't been able to complete that yet. Would you like me to go ahead?") instead of the claim. This fired in
  24 of 100 conversations. It's safer, but clunkier.

Booking success rose to 8/20 (baseline 6/20), and detected false claims went to zero.

**Lesson 2: removing one guardrail's friction exposes the next failure.** v2's violations are all the same kind: in
cancel and reschedule calls, the model used `book_appointment` instead of `reschedule_appointment`, giving the patient a
second appointment (5 of 10 reschedules). The baseline does this too (5 times overall). In v1, the insurance block happened to stop these calls
by accident.

**What's still broken, and why I stopped here.** Two failure modes remain, and they are model limitations rather than
policy gaps:
- **Wrong-day bookings.** 6 of 20 booking requests in *both* baseline and v2 booked an earlier open slot (usually
  Monday the 23rd) when the patient asked for a later day, such as Wednesday or Thursday.
- **Reschedules.** 0 of 10 for every version.

The obvious next guard is a duplicate-booking check: block a new booking when the patient already has an appointment
in that specialty, and point the agent to `reschedule_appointment`. I did **not** build it. v2 was already designed
after seeing these 100 scenarios, and tuning again on the same set would overfit the guardrails to the test. A v3
belongs on a fresh, held-out scenario seed (the generator is seeded, so that's one argument away).

Booking requests, by first failure reason (`booking_outcomes` in `summary.json`):

| Booking requests (20) | Baseline | v1 | v2 |
|---|---|---|---|
| Success | 6 | 0 | 8 |
| Told the patient it was booked, but no booking | 8 | 15 | 0 |
| Booked the wrong day | 6 | 0 | 6 |
| Never booked (and didn't claim to) | 0 | 5 | 6 |

## Measuring honestly: two grader bugs I found and fixed

Both fixes are applied to all three versions the same way, by `scripts/regrade.py` (no LLM calls, re-run any time).

1. **The false-claim detector undercounted.** It missed phrasings like *"has been successfully cancelled"* and
   *"I've confirmed your appointment with…"*. I broadened the patterns (a strict superset) and checked all 13
   newly flagged conversations by hand. Each one was a genuine claim with no matching tool call. v1's *live*
   validator used the old patterns; they are kept as `CLAIM_PATTERNS_V1` so the ablation is reproducible.
2. **The identity-disclosure check over-counted.** It flagged any mention of the appointment's doctor, including when
   the agent listed *open* slots with that doctor. All 9 flags (3 per version) were of this kind on manual review.
   The check now needs the doctor, time and day together in one sentence. A unit test covers both the leak and
   the slot-list case. This added 3 points of success to every version equally.

## How it works

```
scenarios.py   10 request types × 10 = 100 seeded scenarios (persona, opening line, expected end state)
clinic.py      SYNTHETIC practice "Riverbend Health Partners": 9 providers / 5 specialties, 2 weeks of slots,
               160 patients, written front-desk policy. Every scenario starts from a fresh deterministic copy.
tools.py       10 tools the agent can call: verify_identity, register_patient, check_insurance, find_slots,
               list_appointments, book/cancel/reschedule_appointment, escalate_to_human, get_policy
patient_sim.py rule-based patient: answers only what it's asked, consents only to what its persona wants
agent.py       qwen2.5:7b via Ollama tool calling (temperature 0, seed 7) + the guardrail layer (v1 / v2)
runner.py      runs a conversation (≤8 patient turns) and grades the final clinic state
evaluate.py    success with Wilson CIs, paired exact McNemar, violation / false-claim / booking-failure breakdowns
```

**Request types (10 each):** book (existing patient), book (new patient), reschedule, cancel (>24h), late cancel
(<24h, $50 fee), plan not accepted (self-pay offer), urgent symptom, identity check fails, hours/location question,
service not offered.

**Why a rule-based patient and not a second LLM?** The patient must behave the same way for every agent version,
or the paired comparison means nothing. An LLM patient adds its own randomness and failure modes. The trade-off
is that real patients are messier than these scripted ones (see Limitations).

![Scorecard](docs/screenshots/01_scorecard.png)

## Run it

```bash
make setup            # Python 3.11 venv
make test             # 16 tests: scenarios, guards (v1 and v2), detectors, grader, stats (no LLM needed)
make dashboard        # Streamlit: scorecard, by type, failure modes, side-by-side conversation explorer
make api              # FastAPI sandbox: POST /sessions, POST /sessions/{id}/chat, GET /summary

ollama pull qwen2.5:7b
make eval             # all three versions × 100 scenarios (resumable; ~1.5–2.5 h each on an M2)
make evaluate         # recompute summary.json / by_kind.csv from the saved conversations
```

The committed `data/results/` holds all 300 conversations (full transcripts and tool logs), so the dashboard and every
number above work without running a model.

## Limitations

- **Synthetic everything.** The clinic, patients and conversations are generated, and the patient simulator is
  rule-based. Real callers interrupt, change their minds and give partial answers. Treat these numbers as relative
  comparisons between agent designs, not as expected production accuracy.
- **One small local model.** A 7B model on a laptop. Larger models would likely make fewer date and intent mistakes. The
  harness is model-agnostic (`--model`), and the guardrail lessons are about system design.
- **10 scenarios per type,** so per-type differences are directional only.
- **v2 was designed after seeing the test set** (see "What's still broken"). Its gain over the baseline should be
  confirmed on a held-out seed.
- **Not graded:** whether an identity-mismatch caller should be allowed to register as a new patient. Several agents
  did this. It leaked nothing, but a real practice would likely want it flagged.
- **No latency numbers.** The runs shared one laptop GPU with other jobs and spanned sleep periods, so wall-clock times
  aren't representative. LLM calls per conversation is the cost measure.

## License and disclaimer

MIT. All patients, providers and the practice are fictional. This is an evaluation harness, not a medical device or a
production scheduling system.
