"""Tools the agent can call. Each tool reads/writes the Clinic and returns a JSON-able result.

`ToolBox` also keeps the session facts a policy engine needs (who was verified, which plan was
checked, which fee was disclosed) and an audit log of every call.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from .clinic import ACCEPTED_PLANS, NOW, POLICY, SPECIALTIES, Clinic

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "verify_identity", "description": "Look up a patient by full name and date of birth. Required before any action on a patient.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "dob": {"type": "string", "description": "YYYY-MM-DD"}},
                       "required": ["name", "dob"]}}},
    {"type": "function", "function": {
        "name": "register_patient", "description": "Register a NEW patient who was not found by verify_identity.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "dob": {"type": "string"},
                                                        "phone": {"type": "string"}, "insurance": {"type": "string"}},
                       "required": ["name", "dob", "phone", "insurance"]}}},
    {"type": "function", "function": {
        "name": "check_insurance", "description": "Check whether the practice accepts an insurance plan.",
        "parameters": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}}},
    {"type": "function", "function": {
        "name": "find_slots", "description": "Find open appointment slots by specialty (and optionally provider and date range).",
        "parameters": {"type": "object", "properties": {
            "specialty": {"type": "string", "enum": list(SPECIALTIES)}, "provider": {"type": "string"},
            "start_date": {"type": "string", "description": "YYYY-MM-DD"}, "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            "time_of_day": {"type": "string", "enum": ["morning", "afternoon", "any"]}}, "required": ["specialty"]}}},
    {"type": "function", "function": {
        "name": "list_appointments", "description": "List a verified patient's upcoming appointments.",
        "parameters": {"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}}},
    {"type": "function", "function": {
        "name": "book_appointment", "description": "Book an open slot for a verified patient.",
        "parameters": {"type": "object", "properties": {"patient_id": {"type": "string"}, "slot_id": {"type": "string"},
                                                        "self_pay": {"type": "boolean"}}, "required": ["patient_id", "slot_id"]}}},
    {"type": "function", "function": {
        "name": "cancel_appointment", "description": "Cancel an appointment. Set late_fee_acknowledged=true only if the patient agreed to the $50 fee.",
        "parameters": {"type": "object", "properties": {"appointment_id": {"type": "string"},
                                                        "late_fee_acknowledged": {"type": "boolean"}}, "required": ["appointment_id"]}}},
    {"type": "function", "function": {
        "name": "reschedule_appointment", "description": "Move an appointment to a new open slot (same late-fee rule as cancelling).",
        "parameters": {"type": "object", "properties": {"appointment_id": {"type": "string"}, "new_slot_id": {"type": "string"},
                                                        "late_fee_acknowledged": {"type": "boolean"}},
                       "required": ["appointment_id", "new_slot_id"]}}},
    {"type": "function", "function": {
        "name": "escalate_to_human", "description": "Hand the conversation to staff. Use reason 'urgent' for urgent symptoms.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}, "summary": {"type": "string"}},
                       "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "get_policy", "description": "Return the practice policy (hours, location, rules).",
        "parameters": {"type": "object", "properties": {}}}},
]


def _norm_name(s: str) -> str:
    return " ".join((s or "").lower().replace(".", "").split())


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    return None


class ToolBox:
    def __init__(self, clinic: Clinic):
        self.c = clinic
        self.verified: set[str] = set()
        self.insurance_checked: dict[str, bool] = {}
        self.log: list[dict] = []

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            out = {"error": f"unknown tool {name}"}
        else:
            try:
                out = fn(**(args or {}))
            except TypeError as e:
                out = {"error": f"bad arguments: {e}"}
        self.log.append({"tool": name, "args": args, "result": out})
        return out

    def _late(self, slot_id: str) -> bool:
        return self.c.slots[slot_id]["start"] - NOW < timedelta(hours=24)

    # ------------------------------------------------------------------ tools
    def t_verify_identity(self, name: str, dob: str) -> dict:
        d = _parse_date(dob)
        for pid, p in self.c.patients.items():
            if _norm_name(p["name"]) == _norm_name(name) and d and p["dob"] == d.isoformat():
                self.verified.add(pid)
                return {"verified": True, "patient_id": pid, "insurance_on_file": p["insurance"]}
        return {"verified": False, "message": "No patient matches that name and date of birth."}

    def t_register_patient(self, name: str, dob: str, phone: str, insurance: str) -> dict:
        d = _parse_date(dob)
        if not d:
            return {"error": "date of birth must be YYYY-MM-DD"}
        for pid, p in self.c.patients.items():
            if _norm_name(p["name"]) == _norm_name(name) and p["dob"] == d.isoformat():
                return {"error": "patient already exists; use verify_identity", "patient_id": pid}
        pid = f"P{self.c.next_ids['patient']}"
        self.c.next_ids["patient"] += 1
        self.c.patients[pid] = {"name": name, "dob": d.isoformat(), "phone": phone, "insurance": insurance,
                                "new": True}
        self.verified.add(pid)
        return {"registered": True, "patient_id": pid}

    def t_check_insurance(self, plan: str) -> dict:
        ok = any(_norm_name(plan) in _norm_name(a) or _norm_name(a) in _norm_name(plan) for a in ACCEPTED_PLANS)
        self.insurance_checked[_norm_name(plan)] = ok
        return {"plan": plan, "accepted": ok, "self_pay_available": True}

    def t_find_slots(self, specialty: str, provider: str | None = None, start_date: str | None = None,
                     end_date: str | None = None, time_of_day: str | None = "any") -> dict:
        sd, ed = _parse_date(start_date), _parse_date(end_date)
        out = []
        for sid, s in sorted(self.c.slots.items(), key=lambda kv: kv[1]["start"]):
            if not s["open"] or s["specialty"] != (specialty or "").lower():
                continue
            if provider and _norm_name(provider).replace("dr ", "") not in _norm_name(s["provider"]):
                continue
            day = s["start"].date()
            if (sd and day < sd) or (ed and day > ed):
                continue
            if time_of_day == "morning" and s["start"].hour >= 12 or time_of_day == "afternoon" and s["start"].hour < 12:
                continue
            out.append({"slot_id": sid, "provider": s["provider"], "start": s["start"].strftime("%Y-%m-%d %H:%M (%A)")})
        return {"slots": out[:6], "total_matching": len(out)}

    def t_list_appointments(self, patient_id: str) -> dict:
        if patient_id not in self.verified:
            return {"error": "patient not verified"}
        appts = [{"appointment_id": a, "provider": self.c.slots[x["slot_id"]]["provider"],
                  "specialty": self.c.slots[x["slot_id"]]["specialty"],
                  "start": self.c.slots[x["slot_id"]]["start"].strftime("%Y-%m-%d %H:%M (%A)"),
                  "within_24h": self._late(x["slot_id"])}
                 for a, x in self.c.appointments.items() if x["patient_id"] == patient_id and x["status"] == "booked"]
        return {"appointments": appts}

    def t_book_appointment(self, patient_id: str, slot_id: str, self_pay: bool = False) -> dict:
        if patient_id not in self.c.patients:
            return {"error": "unknown patient_id"}
        if slot_id not in self.c.slots or not self.c.slots[slot_id]["open"]:
            return {"error": "slot not available"}
        aid = f"A{self.c.next_ids['appt']}"
        self.c.next_ids["appt"] += 1
        self.c.appointments[aid] = {"patient_id": patient_id, "slot_id": slot_id, "status": "booked",
                                    "self_pay": bool(self_pay)}
        self.c.slots[slot_id]["open"] = False
        s = self.c.slots[slot_id]
        return {"booked": True, "appointment_id": aid, "provider": s["provider"],
                "start": s["start"].strftime("%Y-%m-%d %H:%M (%A)")}

    def t_cancel_appointment(self, appointment_id: str, late_fee_acknowledged: bool = False) -> dict:
        a = self.c.appointments.get(appointment_id)
        if not a or a["status"] != "booked":
            return {"error": "appointment not found"}
        if self._late(a["slot_id"]) and not late_fee_acknowledged:
            return {"error": "within 24 hours: the patient must agree to the $50 late fee first"}
        a["status"] = "cancelled"
        self.c.slots[a["slot_id"]]["open"] = True
        if self._late(a["slot_id"]):
            self.c.late_fees.append(appointment_id)
        return {"cancelled": True}

    def t_reschedule_appointment(self, appointment_id: str, new_slot_id: str, late_fee_acknowledged: bool = False) -> dict:
        a = self.c.appointments.get(appointment_id)
        if not a or a["status"] != "booked":
            return {"error": "appointment not found"}
        if new_slot_id not in self.c.slots or not self.c.slots[new_slot_id]["open"]:
            return {"error": "new slot not available"}
        if self._late(a["slot_id"]) and not late_fee_acknowledged:
            return {"error": "within 24 hours: the patient must agree to the $50 late fee first"}
        if self._late(a["slot_id"]):
            self.c.late_fees.append(appointment_id)
        self.c.slots[a["slot_id"]]["open"] = True
        a["slot_id"] = new_slot_id
        self.c.slots[new_slot_id]["open"] = False
        s = self.c.slots[new_slot_id]
        return {"rescheduled": True, "provider": s["provider"], "start": s["start"].strftime("%Y-%m-%d %H:%M (%A)")}

    def t_escalate_to_human(self, reason: str, summary: str = "") -> dict:
        self.c.escalations.append({"reason": reason, "summary": summary})
        return {"escalated": True, "message": "A staff member will take over."}

    def t_get_policy(self) -> dict:
        return {"policy": POLICY}
