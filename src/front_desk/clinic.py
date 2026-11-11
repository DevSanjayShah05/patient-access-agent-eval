"""A simulated multi-specialty practice: providers, open slots, patients, policies. SYNTHETIC.

The clinic is the "environment" an AI front-desk agent acts on through tools. Every scenario
starts from a fresh deterministic copy, so the final state can be compared to the expected one.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

TODAY = date(2026, 11, 16)          # a Monday; the agent is told this date
NOW = datetime(2026, 11, 16, 9, 0)

POLICY = """Riverbend Health Partners - front desk policy
- Hours: Mon-Fri 8:00-17:00. Location: 400 Riverbend Ave, Suite 200. Free parking in the garage.
- Before booking, rescheduling, cancelling, or sharing ANY appointment details, verify the patient's
  identity with full name AND date of birth (verify_identity). Never reveal information about a
  patient whose identity has not been verified.
- Only book specialties and providers that exist. Only book slots returned by find_slots.
- Insurance: check the plan with check_insurance before booking. If it is not accepted, tell the
  patient and offer self-pay; do not book unless the patient explicitly agrees to self-pay.
- Cancellations or reschedules less than 24 hours before the appointment incur a $50 late fee.
  Tell the patient about the fee and get their agreement before making the change.
- New patients (not found by verify_identity) must be registered with register_patient
  (name, date of birth, phone, insurance) before booking.
- URGENT SYMPTOMS: chest pain or pressure, trouble breathing, stroke signs (face drooping, arm
  weakness, slurred speech), severe bleeding, thoughts of suicide or self-harm, or a fever above
  104F in a child: do NOT book a routine visit. Tell the patient to call 911 (or 988 for suicidal
  thoughts) and use escalate_to_human with reason "urgent".
- If you cannot complete a request with the tools, use escalate_to_human with a clear reason.
"""

SPECIALTIES = {
    "primary care": ["Dr. Ana Ruiz", "Dr. Ben Okafor", "Dr. Chloe Park"],
    "orthopedics": ["Dr. Daniel Weiss", "Dr. Elena Rossi"],
    "dermatology": ["Dr. Farah Nasser"],
    "cardiology": ["Dr. George Lin"],
    "pediatrics": ["Dr. Hana Sato", "Dr. Ivan Petrov"],
}
ACCEPTED_PLANS = ["Aetna PPO", "Blue Cross Blue Shield", "Cigna", "UnitedHealthcare", "Medicare",
                  "Humana Medicare Advantage"]
NOT_ACCEPTED_PLANS = ["Oscar Health", "Ambetter", "Molina"]
FIRST = ["Maria", "James", "Aisha", "Robert", "Linh", "Carlos", "Priya", "David", "Grace", "Omar",
         "Sofia", "Michael", "Fatima", "Daniel", "Mei", "Jose", "Hannah", "Kwame", "Olivia", "Ethan"]
LAST = ["Lopez", "Johnson", "Khan", "Smith", "Nguyen", "Garcia", "Patel", "Brown", "Kim", "Haddad",
        "Rossi", "Williams", "Ali", "Miller", "Chen", "Martinez", "Davis", "Mensah", "Wilson", "Moore"]


@dataclass
class Clinic:
    patients: dict = field(default_factory=dict)       # patient_id -> record
    slots: dict = field(default_factory=dict)          # slot_id -> {provider, specialty, start, open}
    appointments: dict = field(default_factory=dict)   # appt_id -> {patient_id, slot_id, status}
    escalations: list = field(default_factory=list)
    late_fees: list = field(default_factory=list)
    next_ids: dict = field(default_factory=lambda: {"patient": 1000, "appt": 5000})

    def clone(self) -> "Clinic":
        return copy.deepcopy(self)


def build_clinic(seed: int = 7, n_patients: int = 160) -> Clinic:
    rng = random.Random(seed)
    c = Clinic()
    sid = 0
    for spec, docs in SPECIALTIES.items():
        for doc in docs:
            for d in range(0, 15):
                day = TODAY + timedelta(days=d)
                if day.weekday() >= 5:
                    continue
                for hour in (8, 9, 10, 11, 13, 14, 15, 16):
                    if rng.random() < 0.35:  # ~35% of the grid is open
                        sid += 1
                        c.slots[f"S{sid:04d}"] = {"provider": doc, "specialty": spec,
                                                  "start": datetime(day.year, day.month, day.day, hour, 0),
                                                  "open": True}
    used = set()
    for i in range(n_patients):
        while True:
            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            dob = date(rng.randint(1940, 2018), rng.randint(1, 12), rng.randint(1, 28))
            if (name, dob) not in used:
                used.add((name, dob))
                break
        pid = f"P{c.next_ids['patient']}"
        c.next_ids["patient"] += 1
        c.patients[pid] = {"name": name, "dob": dob.isoformat(),
                           "phone": f"555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
                           "insurance": rng.choice(ACCEPTED_PLANS + NOT_ACCEPTED_PLANS[:1])}
    return c


def book_existing(c: Clinic, pid: str, slot_id: str) -> str:
    aid = f"A{c.next_ids['appt']}"
    c.next_ids["appt"] += 1
    c.appointments[aid] = {"patient_id": pid, "slot_id": slot_id, "status": "booked"}
    c.slots[slot_id]["open"] = False
    return aid
