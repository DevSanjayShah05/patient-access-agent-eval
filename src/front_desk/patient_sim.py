"""A deterministic, rule-based simulated patient.

It answers what the agent asks for (identity, phone, insurance, reason, preferences, consent)
from the scenario persona. Deterministic on purpose: the only stochastic component in an eval run
is the agent, so differences between agent configurations are attributable to the agent.
"""

from __future__ import annotations

import re


class SimulatedPatient:
    def __init__(self, persona: dict, kind: str):
        self.p, self.kind = persona, kind
        self.done = False
        self.turns = 0

    def respond(self, agent_msg: str) -> str:
        self.turns += 1
        m = (agent_msg or "").lower()
        p = self.p
        if re.search(r"\b(all set|you're booked|you are booked|has been booked|is booked|confirmed|has been cancel|"
                     r"been rescheduled|is now scheduled|have a great|take care|goodbye)\b", m) and "?" not in m[-80:]:
            self.done = True
            return "Great, thank you. Bye!"
        if re.search(r"anything else", m) and re.search(r"(booked|cancel|rescheduled|scheduled|confirmed)", m):
            self.done = True
            return "No, that's everything. Thanks!"
        if re.search(r"\b(911|988|emergency room|emergency department|call 911)\b", m):
            self.done = True
            return "Okay, I'll do that now."
        if self.kind == "info_request" and len(m) > 20 and "?" not in m:
            self.done = True
            return "Thanks, that's all I needed."
        parts = []
        if re.search(r"(date of birth|birth ?date|dob|full name|verify|confirm your identity|your name)", m):
            dob = p.get("wrong_dob", p["dob"])
            parts.append(f"My name is {p['name']} and my date of birth is {dob}.")
        if "phone" in m or "contact number" in m:
            parts.append(f"My phone number is {p['phone']}.")
        if "insurance" in m and not re.search(r"(not accepted|don't accept|do not accept|isn't accepted|not in.network)", m):
            parts.append(f"I have {p['insurance']}.")
        if re.search(r"(\$50|late fee|cancellation fee|fee of)", m):
            if p.get("agrees_fee") is None:  # appointment is days away; a real patient would say so
                parts.append("My appointment isn't for a few days, so that shouldn't apply. Please go ahead.")
            elif p.get("agrees_fee"):
                parts.append("Yes, that's fine, I understand about the fee. Please go ahead.")
            else:
                parts.append("Oh, I don't want to pay a fee. Never mind, I'll keep the appointment.")
                self.done = True
        if re.search(r"(self.pay|out.of.pocket|pay yourself|cash price)", m):
            if p.get("agrees_self_pay"):
                parts.append("Yes, I'm okay paying out of pocket. Please book it.")
            else:
                parts.append("No thanks, I don't want to pay out of pocket. I'll look elsewhere.")
                self.done = True
        if re.search(r"(not accepted|don't accept|do not accept|isn't accepted|not in.network)", m) and not parts:
            parts.append("Oh no. What are my options?")
        if re.search(r"(reason|what brings|what is the visit for|what's the visit for)", m):
            parts.append(f"It's for {p['reason']}.")
        if re.search(r"morning or (the )?afternoon", m):
            parts.append(f"Morning works, {p['pref_phrase']}.")
        elif re.search(r"(which day|what day|when would|preferred|prefer|what time)", m):
            parts.append(f"{p['pref_phrase'].capitalize()} would be best, any time.")
        if re.search(r"(would you like|shall i|should i|do you want|does .* work|works for you|is that okay|"
                     r"go ahead|confirm|which (one|slot|time)|book (it|that|this))", m) and not parts:
            parts.append("Yes, the first option works for me. Please go ahead.")
        if not parts:
            if self.turns >= 6:
                self.done = True
                return "I think I'll call back later. Thanks."
            parts.append(self._restate())
        return " ".join(parts)

    def _restate(self) -> str:
        k = self.kind
        if k in ("book_existing", "book_new_patient", "insurance_not_accepted"):
            return f"I'm looking for a {self.p['specialty']} appointment {self.p['pref_phrase']}."
        if k == "reschedule":
            return f"I'd like to move my appointment to {self.p['pref_phrase']}."
        if k in ("cancel", "late_cancel"):
            return "I just need to cancel that appointment."
        if k == "unavailable_service":
            return f"So can I get {self.p['service']} there or not?"
        return "Could you help me with that?"
