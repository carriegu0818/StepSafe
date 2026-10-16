# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Parse the monitor's per-step verdict.

The monitor ends each response with one brace-balanced ``\boxed{...}``:

    \boxed{act:CONTINUE|ASK|STOP, risk:0-100, obj:<entity|NONE>, fail:<label>,
           fault:<label>, harm:<label>, src:<label>, timing:<label>,
           q:<question|NONE>, cite:<sequence|NONE>}

``parse_decision`` tolerates missing keys, a missing backslash and extra prose
(it also reads legacy keys such as ``chan``). ``valid`` is True iff a usable
``act`` was found.
"""

from __future__ import annotations

import re
from typing import List, Tuple

ACTS = ("CONTINUE", "ASK", "STOP")
BINARY_LABELS = ("SAFE", "UNSAFE")
V_CHANNELS = ("V-OT", "V-OR", "V-IC", "V-ID")
A_CHANNELS = ("A-INJECT", "A-CONSENT", "A-LEAK", "A-STATE", "A-POLICY", "A-HARM", "A-GEN")
CHANNELS = V_CHANNELS + A_CHANNELS
PERTURBATION_TYPES = ("P-II", "P-AG", "P-RB")
_ACT_ALIASES = {
    "PASS": "CONTINUE", "OK": "CONTINUE", "ALLOW": "CONTINUE", "WAIT": "CONTINUE",
    "CLARIFY": "ASK", "QUESTION": "ASK",
    # BLOCK remains accepted for older checkpoints, but STOP is canonical.
    "BLOCK": "STOP", "HALT": "STOP", "DENY": "STOP",
    "ALERT": "STOP", "ALARM": "STOP", "FLAG": "STOP",
}
_RISK_WORD_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "one hundred": 100,
}


def extract_boxed(text: str) -> str | None:
    r"""Return the content of the LAST brace-balanced ``\boxed{...}``, else None."""
    if not text:
        return None
    starts = [m.end() for m in re.finditer(r"\\?boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    depth, i = 1, start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return text[start:]


def _norm_act(raw: str) -> str | None:
    v = (raw or "").strip().upper()
    if v in ACTS:
        return v
    return _ACT_ALIASES.get(v)


def _norm_binary(raw: str) -> str | None:
    value = (raw or "").strip().upper()
    if value in BINARY_LABELS:
        return value
    if value in ("0", "FALSE"):
        return "SAFE"
    if value in ("1", "TRUE"):
        return "UNSAFE"
    return None


def _norm_chan(raw: str) -> str | None:
    v = (raw or "").strip().upper()
    if not v or v in ("NONE", "N/A", "NA"):
        return None
    mc = re.search(r"\bV[\-_ ]?(OT|OR|IC|ID)\b", v)
    if mc:
        return "V-" + mc.group(1)
    mc = re.search(r"\bA[\-_ ]?(INJECT|CONSENT|LEAK|STATE|POLICY|HARM|GEN)\b", v)
    if mc:
        return "A-" + mc.group(1)
    return None


def _norm_pert(raw: str) -> str | None:
    v = (raw or "").strip().upper()
    if not v or v in ("NONE", "N/A", "NA"):
        return None
    m = re.search(r"\bP[\-_ ]?(II|AG|RB)\b", v)
    return f"P-{m.group(1)}" if m else None


def _unwrap_latex_text(content: str) -> str:
    """Remove a whole-decision LaTeX text wrapper before field parsing."""
    content = re.sub(r"\s*\\[)\]]\s*$", "", content)
    match = re.fullmatch(r"\s*\\(?:text|textrm|mathrm)\s*\{(.*)\}\s*", content, re.DOTALL)
    content = match.group(1).strip() if match else content
    # Models sometimes use TeX spacing before a field name, e.g. ``\ act``.
    # It is presentation-only and must not turn a valid boxed field invalid.
    content = re.sub(r"\\(?=\s)", "", content)
    content = re.sub(r"\\(?:quad|qquad|enspace|thinspace)\b", " ", content)
    content = re.sub(r"\\[,;!]", "", content)
    return content


def _parse_risk(raw: str) -> int | None:
    number = re.search(r"-?\d+(\.\d+)?", raw)
    if number:
        # Clamp before int(): a runaway digit string (>308 digits) parses to inf, and
        # int(round(inf)) raises OverflowError; clamping first is identical for finite values.
        return int(round(max(0.0, min(100.0, float(number.group(0))))))
    words = " ".join(re.findall(r"[a-z]+", raw.lower()))
    tokens = words.split()
    for index, token in enumerate(tokens[:-1]):
        tens = _RISK_WORD_VALUES.get(token)
        units = _RISK_WORD_VALUES.get(tokens[index + 1])
        if tens is not None and units is not None and 20 <= tens <= 90 and 0 < units < 10:
            return tens + units
    for phrase, value in sorted(_RISK_WORD_VALUES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", words):
            return value
    return None


def parse_decision(text: str) -> dict:
    """Parse one boxed decision into current action and binary projection."""
    out = {"risk": None, "act": None, "binary": None, "chan": None, "pert": None,
           "obj": None, "q": None,
           # Taxonomy fields of the verdict; recorded, never required.
           "fail": None, "harm": None, "src": None,
           "fault": None, "timing": None,
           # cite: the earlier record that justifies the verdict; optional.
           "cite": None,
           "valid": False, "raw": ""}
    content = extract_boxed(text)
    if content is None:
        return out
    content = _unwrap_latex_text(content)
    out["raw"] = content.strip()
    # split on top-level commas/semicolons/newlines; keep simple key:value tokens
    fields: dict[str, str] = {}
    for tok in re.split(r"[,\n;]+", content):
        m = re.match(r"\s*([A-Za-z_]+)\s*[:=]\s*(.*\S)?\s*$", tok)
        if not m:
            continue
        key = m.group(1).strip().lower()
        val = (m.group(2) or "").strip()
        if key not in fields:
            fields[key] = val
    # act
    act = _norm_act(
        fields.get("act")
        or fields.get("action")
        or fields.get("decision")
        or fields.get("step")
        or ""
    )
    out["act"] = act
    out["binary"] = _norm_binary(
        fields.get("binary")
        or fields.get("binary_label")
        or fields.get("label")
        or ""
    )
    # risk score 0-100
    rraw = fields.get("risk") or fields.get("score")
    if rraw is not None:
        out["risk"] = _parse_risk(rraw)
    # violation channel; tolerant of "VID"/"v_id"/"V-ID" and A_* forms.
    chan_raw = fields.get("chan") or fields.get("channel") or fields.get("code")
    out["chan"] = _norm_chan(chan_raw or "")
    pert_raw = fields.get("pert") or fields.get("perturbation") or fields.get("stressor")
    out["pert"] = _norm_pert(pert_raw or "")
    # Taxonomy fields: stored verbatim (lowercased), with NONE folded to None.
    # Absence is not an error and never affects `valid`.
    for k_src, k_dst in (("fail", "fail"), ("failure_mode", "fail"),
                         ("harm", "harm"), ("harm_type", "harm"),
                         ("src", "src"), ("risk_source", "src"),
                         ("fault", "fault"), ("timing", "timing")):
        if out[k_dst] is None:
            raw = (fields.get(k_src) or "").strip()
            if raw and raw.upper() != "NONE":
                out[k_dst] = raw.lower()
    # object + question (NONE -> None)
    for k_src, k_dst in (("obj", "obj"), ("object", "obj"), ("q", "q"), ("question", "q")):
        if k_src in fields and out[k_dst] is None:
            v = fields[k_src].strip().strip('"').strip("'")
            out[k_dst] = None if v.upper() in ("NONE", "N/A", "") else v
    craw = fields.get("cite") or fields.get("evidence_step") or fields.get("because")
    if craw:
        m = re.search(r"-?\d+", str(craw))
        if m:
            try:
                out["cite"] = int(m.group(0))
            except ValueError:
                out["cite"] = None
    out["valid"] = act is not None
    return out


def stepsafe_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Env-manager projection hook for a batch of raw model texts.

    The replay env reparses the raw text in ``step``. This hook only marks whether
    the response contains a usable boxed verdict (``is_action_valid``).
    """
    results: List[str] = []
    valids: List[int] = [1] * len(actions)
    for i, action in enumerate(actions):
        action = action if isinstance(action, str) else ""
        results.append(action)
        if parse_decision(action).get("valid"):
            continue
        else:
            valids[i] = 0
    return results, valids
