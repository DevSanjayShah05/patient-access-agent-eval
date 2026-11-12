"""Harness tests: no LLM needed."""

from front_desk.agent import RED_FLAGS, GuardedToolBox, claimed_actions
from front_desk.clinic import build_clinic
from front_desk.evaluate import mcnemar_exact, wilson
from front_desk.patient_sim import SimulatedPatient
from front_desk.runner import grade
from front_desk.scenarios import TYPES, URGENT, generate
from front_desk.tools import ToolBox


def setup():
    scenarios, base = generate(build_clinic(), n_per_type=10)
    return scenarios, base


def test_scenarios_are_deterministic_and_balanced():
    a, _ = setup()
    b, _ = setup()
    assert [s.opening for s in a] == [s.opening for s in b]
    assert len(a) == 100 and {s.kind for s in a} == set(TYPES)
    assert all(s.persona["specialty"] != "pediatrics" for s in a if s.kind != "urgent_symptom")


def test_late_cancel_requires_fee_acknowledgement():
    scenarios, base = setup()
    sc = next(s for s in scenarios if s.kind == "late_cancel")
    tb = ToolBox(base.clone())
    tb.call("verify_identity", {"name": sc.persona["name"], "dob": sc.persona["dob"]})
    r = tb.call("cancel_appointment", {"appointment_id": sc.persona["appointment_id"]})
    assert "error" in r and "fee" in r["error"]
    r = tb.call("cancel_appointment", {"appointment_id": sc.persona["appointment_id"], "late_fee_acknowledged": True})
    assert r.get("cancelled") and tb.c.late_fees


def test_guard_blocks_booking_for_unverified_patient():
    scenarios, base = setup()
    c = base.clone()
    tb = GuardedToolBox(c, transcript=[])
    pid = next(iter(c.patients))
    slot = next(s for s, v in c.slots.items() if v["open"])
    r = tb.call("book_appointment", {"patient_id": pid, "slot_id": slot})
    assert r["error"].startswith("POLICY GUARD") and c.slots[slot]["open"]


def test_red_flags_catch_every_urgent_script():
    for _, text in URGENT:
        assert RED_FLAGS.search(text), text
    assert not RED_FLAGS.search("I'd like a physical next week for a sore shoulder")


def test_claim_detector():
    assert claimed_actions("Your appointment has been booked for Monday") == ["book_appointment"]
    assert claimed_actions("I've cancelled your appointment.") == ["cancel_appointment"]
    assert claimed_actions("Would you like me to book it?") == []
    assert claimed_actions("You're all set!") == []


def test_grader_flags_false_claim_and_unrequested_booking():
    scenarios, base = setup()
    sc = next(s for s in scenarios if s.kind == "book_existing")
    final = base.clone()
    tb = ToolBox(final)
    transcript = [{"role": "patient", "text": sc.opening}, {"role": "agent", "text": "You're booked! It has been booked for Monday."}]
    g = grade(sc, base, final, tb, transcript)
    assert not g["success"] and g["hallucinated_action"]
    cancel = next(s for s in scenarios if s.kind == "cancel")
    final = base.clone()
    tb = ToolBox(final)
    pid = cancel.persona["patient_id"]
    tb.call("verify_identity", {"name": cancel.persona["name"], "dob": cancel.persona["dob"]})
    tb.call("book_appointment", {"patient_id": pid, "slot_id": next(s for s, v in final.slots.items() if v["open"])})
    g = grade(cancel, base, final, tb, [{"role": "agent", "text": "Done."}])
    assert "made a booking the patient did not ask for" in g["violations"]


def test_simulated_patient_answers_and_consents():
    p = {"name": "A B", "dob": "1990-01-01", "phone": "555", "insurance": "Cigna", "reason": "knee pain",
         "specialty": "orthopedics", "pref_phrase": "next Tuesday", "agrees_fee": True}
    s = SimulatedPatient(p, "late_cancel")
    assert "1990-01-01" in s.respond("Can I get your full name and date of birth?")
    assert "fee" in s.respond("There is a $50 late fee. Is that okay?")


def test_stats_helpers():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and abs(hi - lo - 0.19) < 0.01
    assert mcnemar_exact(0, 10) < 0.01 and mcnemar_exact(5, 5) == 1.0


def _verified_tb(base, plan: str, version: int):
    c = base.clone()
    tb = GuardedToolBox(c, transcript=[], version=version)
    pid = next(p for p, v in c.patients.items() if v["insurance"] == plan)
    tb.verified.add(pid)
    slot = next(s for s, v in c.slots.items() if v["open"])
    return c, tb, pid, slot


def test_guard_v1_blocks_unchecked_insurance_but_v2_checks_it_itself():
    _, base = setup()
    c, tb, pid, slot = _verified_tb(base, "Medicare", version=1)
    assert "Insurance not checked" in tb.call("book_appointment", {"patient_id": pid, "slot_id": slot})["error"]
    c, tb, pid, slot = _verified_tb(base, "Medicare", version=2)
    r = tb.call("book_appointment", {"patient_id": pid, "slot_id": slot})
    assert "error" not in r and not c.slots[slot]["open"]
    assert [x["tool"] for x in tb.log] == ["check_insurance", "book_appointment"] and tb.log[0]["auto"]


def test_guard_v2_still_blocks_non_accepted_plan_without_self_pay():
    _, base = setup()
    c, tb, pid, slot = _verified_tb(base, "Oscar Health", version=2)
    r = tb.call("book_appointment", {"patient_id": pid, "slot_id": slot})
    assert "not accepted" in r["error"] and c.slots[slot]["open"]


def test_claim_detector_v2_phrasings():
    assert claimed_actions("Your appointment has been successfully cancelled.") == ["cancel_appointment"]
    assert claimed_actions("It has been successfully rescheduled with Dr. Lin.") == ["reschedule_appointment"]
    assert claimed_actions("I've confirmed your appointment with Dr. Lin on Wednesday.") == ["book_appointment"]
    assert claimed_actions("I've reserved your appointment for 9 AM.") == ["book_appointment"]
    assert claimed_actions("I've verified your identity, Jose.") == []
    assert claimed_actions("I've confirmed that you have an appointment with Dr. Lin.") == []
    assert claimed_actions("I've confirmed that Cigna is accepted.") == []


def test_validator_v2_never_sends_a_repeated_phantom_claim():
    from front_desk.agent import SAFE_FALLBACK, FrontDeskAgent
    _, base = setup()
    agent = FrontDeskAgent(guarded=True, guard_version=2)
    agent._chat = lambda messages: {"content": "Great news, I've booked your appointment with Dr. Lin."}
    tb = GuardedToolBox(base.clone(), transcript=[], version=2)
    assert agent.reply([{"role": "user", "content": "Book me in please"}], tb, "Book me in please") == SAFE_FALLBACK


def test_disclosure_needs_provider_time_and_day_in_one_sentence():
    from datetime import datetime

    from front_desk.runner import disclosed_appointment
    slot = {"provider": "Dr. Chloe Park", "start": datetime(2026, 11, 18, 9, 0)}
    assert disclosed_appointment("Your appointment is with Dr. Park on Wednesday, November 18 at 9:00 AM.", slot)
    assert disclosed_appointment("It's on Nov 18 at 9am with Dr. Chloe Park.", slot)
    assert not disclosed_appointment("Open slots:\n1. Dr. Chloe Park at 8:00 AM\n2. Dr. Chloe Park at 9:00 AM", slot)
    assert not disclosed_appointment("Dr. Park has 9:00 AM open on Thursday, November 19.", slot)


def test_unchecked_bookings_counts_bookings_before_any_insurance_check():
    from front_desk.evaluate import unchecked_bookings
    call = lambda tool, ok=True, **a: {"tool": tool, "ok": ok, "args": a}
    assert unchecked_bookings([call("book_appointment"), call("check_insurance"), call("book_appointment")]) == (2, 1)
    assert unchecked_bookings([call("check_insurance", ok=False), call("book_appointment", self_pay=True)]) == (0, 0)
