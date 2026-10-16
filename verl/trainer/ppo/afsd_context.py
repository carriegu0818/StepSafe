"""Teacher context of Adaptive Feedback Self-Distillation (AFSD), Section 2.4 of the paper.

The teacher shares the student's parameters and scores the student's sampled tokens
under a training-only context rho_t:

* Intervention step (annotated verdict ASK or STOP): the annotated ten-field verdict
  l*_t and a description e_t of the violation at this step ("taxonomy and timing
  feedback").
* Safe step (annotated verdict CONTINUE): the CONTINUE verdict, the annotated reason
  to continue when one exists, a note that the student's claimed failure is not
  supported (only when the student intervened), and up to two rules retrieved from
  the false-alarm rubric B.

The false-alarm rubric is mined from the student's own false alarms during training:

1. Every intervention the student issues on a safe step is appended to the false-alarm
   log L with its key kappa = (claimed failure, action surface).
2. Every ``evolve_every`` teacher calls, each key whose log group holds at least
   ``min_cluster`` cases from at least two trajectories receives a candidate rule
   (condition, rationale), written by the live model.
3. Admission rule: the same model, applying the rule literally, must clear at
   least half of up to three source cases (distinct trajectories) and none of up to five
   known violations with the same key (the most similar one, the unsafe counterparts of
   benign twins, then random ones). A rule that clears a violation is rewritten once with
   that violation as a counterexample and tested again.
4. Admitted rules are retrieved by the student's claim with an upper confidence bound
   whose reward is the drop in false alarms on the rule's key.

Nothing about later steps reaches the teacher: no trajectory label and no landmarks.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_system.environments.prompts.stepsafe import FIELDS, TAXONOMY

TAXONOMY_FIELDS = ("fail", "harm", "src", "fault", "timing")
# Turns presented for a verdict that are not agent actions (e.g. a ``policy`` turn of a
# red-teamed row): nothing can be blocked there, so the reference is CONTINUE / all NONE.
NON_ACTION_ROLES = ("policy", "user", "environment", "system")
# Provenance states accepted for a step annotation.
VERIFIED_STATUSES = (
    "source_verified_existing_sft_audit",
    "main_assistant_labels_only",
    "exact_visible_prefix_reuse",
    "source_exempt_safe",
    "source_preserved_out_of_scope",
)


def _norm_label(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return "NONE" if not text or text.upper() in ("NONE", "NULL", "N/A") else text


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _boxed(box: dict) -> str:
    return "\\boxed{" + ", ".join(f"{k}:{box[k]}" for k in FIELDS) + "}"


def non_action_box() -> dict:
    return {k: (0 if k == "risk" else "CONTINUE" if k == "act" else "NONE") for k in FIELDS}


# --------------------------------------------------------------------- annotations
class StepAnnotations:
    """Annotated ten-field verdict of every step, keyed by (trajectory id, sequence)."""

    def __init__(self, path: str):
        self.records: Dict[Tuple[str, int], dict] = {}
        with open(path) as stream:
            for line in stream:
                record = json.loads(line)
                key = (record["audit_id"], record["sequence_num"])
                if key in self.records:
                    raise ValueError(f"Duplicate step annotation: {key}")
                box = record["box"]
                if tuple(box) != FIELDS:
                    raise ValueError(f"Incorrect ten-field order: {key}")
                if record["status"] not in VERIFIED_STATUSES:
                    raise ValueError(f"Unverified step annotation: {key}")
                if record["status"] == "exact_visible_prefix_reuse" and not all(
                    record.get(k) for k in ("source_audit_id", "source_sequence_num", "visible_sha256")
                ):
                    raise ValueError(f"Missing reuse provenance: {key}")
                if box["act"] not in ("CONTINUE", "ASK", "STOP"):
                    raise ValueError(f"Invalid action: {key}")
                if type(box["risk"]) is not int or not 0 <= box["risk"] <= 100:
                    raise ValueError(f"Invalid risk: {key}")
                for field, values in TAXONOMY.items():
                    if box[field] not in {*values, "NONE", "UNKNOWN"}:
                        raise ValueError(f"Invalid {field}: {key}")
                if (box["act"] == "ASK") == (box["q"] == "NONE"):
                    raise ValueError(f"Invalid question: {key}")
                if box["cite"] != "NONE" and not 0 < int(box["cite"]) < key[1]:
                    raise ValueError(f"Non-prior citation: {key}")
                response = record["response"]
                if not response.rstrip().endswith(_boxed(box)):
                    raise ValueError(f"Response disagrees with box: {key}")
                if response.strip() != _boxed(box) and "</think>" not in response:
                    raise ValueError(f"Unexpected reference text: {key}")
                self.records[key] = record
        if not self.records:
            raise ValueError("Empty step annotation file")

    def missing_is_safe(self, key, event_role=None, safe_trajectory=None) -> bool:
        """An unannotated turn is safe when it is not an agent action, or when its
        trajectory is safe (where every step is scored SAFE)."""
        return key not in self.records and (
            str(event_role or "").lower() in NON_ACTION_ROLES or safe_trajectory is True)


class TrajectoryAnnotations:
    """Per-trajectory annotations (label, landmarks, records, step labels), keyed by id."""

    def __init__(self, path: str):
        self.records: Dict[str, dict] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("audit_id"):
                    self.records[rec["audit_id"]] = rec
        print(f"[AFSD] loaded {len(self.records)} trajectory annotations")


class StepTruth:
    """Annotation of one (trajectory, step) in the verdict vocabulary."""

    __slots__ = ("audit_id", "seq", "verdict", "unsafe_traj", "reference_box", "reason",
                 "context_note", "violation", "stage_lines", "claim_label", "surface", "action_text")

    def __init__(self, **kw):
        for key in self.__slots__:
            setattr(self, key, kw.get(key))

    @property
    def flag(self) -> bool:
        return self.verdict in ("ASK", "STOP")


class GroundTruth:
    def __init__(self, trajectories: TrajectoryAnnotations, steps: StepAnnotations):
        self.trajectories = trajectories
        self.steps = steps
        self._negatives: Optional[List[dict]] = None

    def record(self, audit_id: str) -> dict:
        return self.trajectories.records.get(audit_id or "") or {}

    @staticmethod
    def label(record: dict, seq: Optional[int]) -> dict:
        for lab in record.get("step_labels", []) or []:
            try:
                if int(lab.get("sequence_num", -1)) == seq:
                    return dict(lab)
            except (TypeError, ValueError):
                continue
        return {}

    @staticmethod
    def event(record: dict, seq: Optional[int]) -> dict:
        traj = record.get("trajectory") or []
        if isinstance(traj, str):
            try:
                traj = json.loads(traj)
            except ValueError:
                traj = []
        for ev in traj:
            try:
                if int(ev.get("sequence_num", -1)) == seq:
                    return dict(ev)
            except (TypeError, ValueError):
                continue
        return {}

    @staticmethod
    def action_text(ev: dict, limit: int = 600) -> str:
        parts = [str(ev.get("agent_role") or ""), str(ev.get("surface") or "")]
        if ev.get("tool_name") and str(ev.get("tool_name")) != "None":
            parts.append(f"{ev['tool_name']}({str(ev.get('tool_args') or '')[:200]})")
        if ev.get("target_role") and str(ev.get("target_role")) != "None":
            parts.append(f"-> {ev['target_role']}")
        content = str(ev.get("content") or "").replace("\n", " ").strip()
        text = " ".join(p for p in parts if p) + (f" :: {content}" if content else "")
        return text[:limit]

    @staticmethod
    def surface(ev: dict) -> str:
        surf = str(ev.get("surface") or "").strip().lower()
        return surf if surf and surf != "none" else "action"

    def truth(self, audit_id: str, seq: Optional[int]) -> StepTruth:
        record = self.record(audit_id)
        if not record:
            raise ValueError(f"AFSD: no annotation for audit_id={audit_id!r}")
        lab = self.label(record, seq)
        ev = self.event(record, seq)
        unsafe = bool(record.get("unsafe"))

        stage_lines: List[str] = []
        verdict = str(lab.get("episode_act_target") or "CONTINUE").upper()
        if verdict == "NOT_REACHED":
            verdict = "STOP"
            stage_lines = ["Post-incident: the prevention window was missed; STOP now."]
        t_signal = record.get("t_signal")
        if (unsafe and t_signal is not None and seq is not None and int(seq) < int(t_signal)
                and verdict != "CONTINUE"):
            # Before the signal nothing is visible to intervene on.
            verdict = "CONTINUE"
            stage_lines = []

        ref = self.steps.records.get((audit_id, seq))
        role = str(ev.get("agent_role") or "").lower()
        if ref is None and self.steps.missing_is_safe((audit_id, seq), role, not unsafe):
            reference_box = non_action_box()
            verdict = "CONTINUE"
        elif ref is None:
            raise ValueError(f"AFSD: missing step annotation {(audit_id, seq)}")
        else:
            reference_box = dict(ref["box"])
            verdict = str(reference_box["act"]).upper()

        reason = str(lab.get("audit_reason") or "").strip() or None
        context_note = None
        t_ctx = record.get("t_context_risk")
        if (record.get("context_risk") in (True, "True", "true") and t_ctx is not None
                and seq is not None and int(t_ctx) <= int(seq)):
            context_note = (f"Context risk is visible (object: {record.get('context_risk_object')}), "
                            "but context alone is not a stopping target.")
        violation = None
        for v in lab.get("violations") or []:
            violation = str(v.get("description") or v.get("risk_object") or "").strip() or None
            if violation:
                break
        return StepTruth(
            audit_id=audit_id, seq=seq, verdict=verdict, unsafe_traj=unsafe,
            reference_box=reference_box, reason=reason, context_note=context_note,
            violation=violation, stage_lines=stage_lines,
            claim_label=_norm_label(reference_box.get("fail")),
            surface=self.surface(ev), action_text=self.action_text(ev),
        )

    def negatives(self) -> List[dict]:
        """Known violations (annotated ASK/STOP steps), keyed like a claim would be."""
        if self._negatives is not None:
            return self._negatives
        out: List[dict] = []
        for audit_id, record in self.trajectories.records.items():
            if not record.get("unsafe"):
                continue
            for lab in record.get("step_labels", []) or []:
                try:
                    seq = int(lab.get("sequence_num", -1))
                except (TypeError, ValueError):
                    continue
                ref = self.steps.records.get((audit_id, seq))
                if ref is None or ref["box"]["act"] == "CONTINUE":
                    continue
                claim = _norm_label(ref["box"].get("fail"))
                ev = self.event(record, seq)
                text = self.action_text(ev)
                out.append({"audit_id": audit_id, "seq": seq, "claim": claim.lower(),
                            "surface": self.surface(ev), "text": text, "tokens": _tokens(text),
                            "pair": record.get("contrastive_pair_id") or record.get("perturbation_pair_id") or ""})
        self._negatives = out
        return out


# ------------------------------------------------------------------ false-alarm log
class FalseAlarmLog:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._cache: Optional[List[dict]] = None

    def append(self, rec: dict) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        if self._cache is not None:
            self._cache.append(rec)

    def load(self) -> List[dict]:
        if self._cache is None:
            self._cache = []
            if self.path and os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self._cache.append(json.loads(line))
                            except ValueError:
                                continue
        return self._cache


# ------------------------------------------------------------------- rubric
class FalseAlarmRubric:
    """The rubric B: admitted "clean when" rules, keyed by (claimed fail, surface).

    Retrieval is a UCB bandit over the rules matching a step. A rule's reward is measured
    once per evolution window in which it was shown: 1 minus the student's false-alarm
    rate on the rule's key in that window (false alarms on the key per safe step seen on
    that surface). Rules never shown are tried first.
    """

    def __init__(self, path: Optional[str], max_rules_per_key: int = 3, max_rules: int = 64,
                 ucb_c: float = 0.5):
        self.path = path
        self.max_rules_per_key = int(max_rules_per_key)
        self.max_rules = int(max_rules)
        self.ucb_c = float(ucb_c)
        self.data = {"version": 2, "round": 0, "rules": [], "rejected": 0, "log_cursor": 0,
                     "window": self._empty_window()}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.data.update(json.load(f))
        self.data.setdefault("window", self._empty_window())

    @staticmethod
    def _empty_window() -> dict:
        return {"safe_calls": {}, "false_alarms": {}, "shown": {}}

    @property
    def rules(self) -> List[dict]:
        return self.data["rules"]

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    def for_key(self, claim: str, surface: str) -> List[dict]:
        return [r for r in self.rules if r["claim"] == claim and r["surface"] == surface]

    @staticmethod
    def mean_reward(rule: dict) -> float:
        n = int(rule.get("n", 0))
        return float(rule.get("reward_sum", 0.0)) / n if n else 0.0

    def ucb(self, rule: dict) -> float:
        n = int(rule.get("n", 0))
        if n == 0:
            return float("inf")
        total = max(1, sum(int(r.get("n", 0)) for r in self.rules))
        return self.mean_reward(rule) + self.ucb_c * math.sqrt(math.log(total + 1) / n)

    def _evict_one(self, pool: List[dict]) -> None:
        # Never evict an untried rule while a tried one exists; else the lowest mean reward.
        tried = [r for r in pool if int(r.get("n", 0)) > 0] or pool
        self.rules.remove(min(tried, key=lambda r: (self.mean_reward(r), r["created_round"])))

    def add(self, rule: dict) -> None:
        same = self.for_key(rule["claim"], rule["surface"])
        if len(same) >= self.max_rules_per_key:
            self._evict_one(same)
        if len(self.rules) >= self.max_rules:
            self._evict_one(list(self.rules))
        rule["id"] = f"R{self.data.get('next_id', 1)}"
        rule.setdefault("n", 0)
        rule.setdefault("reward_sum", 0.0)
        self.data["next_id"] = self.data.get("next_id", 1) + 1
        self.rules.append(rule)

    def observe(self, surface: str, false_alarm_claim: Optional[str]) -> None:
        w = self.data["window"]
        w["safe_calls"][surface] = w["safe_calls"].get(surface, 0) + 1
        if false_alarm_claim is not None:
            key = f"{false_alarm_claim}|{surface}"
            w["false_alarms"][key] = w["false_alarms"].get(key, 0) + 1

    def retrieve(self, claim: Optional[str], surface: str, k: int = 2) -> List[dict]:
        """Up to two rules for the student's claim, or one for the surface when it did not intervene."""
        if claim:
            pool = self.for_key(claim, surface)
        else:
            pool = [r for r in self.rules if r["surface"] == surface]
            k = min(k, 1)
        hits = sorted(pool, key=lambda r: (-self.ucb(r), -r["created_round"]))[:k]
        shown = self.data["window"]["shown"]
        for r in hits:
            r["retrieved"] = r.get("retrieved", 0) + 1
            shown[r["id"]] = shown.get(r["id"], 0) + 1
        return hits

    def close_window(self) -> dict:
        """Credit every rule shown in the window, then start a new window."""
        w = self.data["window"]
        credited = {}
        for rule in self.rules:
            if not w["shown"].get(rule["id"]):
                continue
            calls = w["safe_calls"].get(rule["surface"], 0)
            if not calls:
                continue
            fa = w["false_alarms"].get(f"{rule['claim']}|{rule['surface']}", 0)
            reward = 1.0 - min(1.0, fa / calls)
            rule["n"] = int(rule.get("n", 0)) + 1
            rule["reward_sum"] = float(rule.get("reward_sum", 0.0)) + reward
            credited[rule["id"]] = round(reward, 4)
        self.data["window"] = self._empty_window()
        return credited


def format_rule(rule: dict) -> str:
    return (f"{rule['id']} (claim fail={rule['claim']} on a {rule['surface']} step): "
            f"When {rule['condition']}, the step is SAFE; it is not {rule['claim']}. {rule['rationale']}")


# ------------------------------------------------------------------ mining round
WRITER_SYSTEM = (
    "You write one precise rule explaining why a class of agent actions is SAFE even though a "
    "safety monitor flagged it. Use only facts observable at that step (the action, its "
    "arguments, the role, permissions shown earlier). Never refer to later steps. Output only a "
    'JSON object: {"condition": "<when this holds>", "rationale": "<why it is not a violation>"}.'
)
JUDGE_SYSTEM = (
    "You check whether a safety rule applies to agent actions. For each numbered case, decide "
    "whether the rule, applied literally, declares that case SAFE. Output only a JSON list of "
    'YES/NO strings, one per case, e.g. ["YES", "NO"].'
)


def _parse_json(text: str, kind):
    text = str(text or "")
    if "</think>" in text:
        text = text.split("</think>")[-1]
    pattern = r"\{.*\}" if kind is dict else r"\[.*\]"
    m = re.search(pattern, text, flags=re.S)
    if not m:
        return None
    try:
        value = json.loads(m.group(0))
    except ValueError:
        return None
    return value if isinstance(value, kind) else None


def _parse_fields(text: str) -> Optional[dict]:
    """Accept ``condition: ...`` / ``rationale: ...`` lines when the model skips the JSON."""
    tail = str(text or "").split("</think>")[-1]
    out = {}
    for key in ("condition", "rationale"):
        m = re.search(rf"{key}\s*[:=]\s*(.+)", tail, flags=re.I)
        if m:
            out[key] = m.group(1).strip().strip('",')
    return out or None


def _deterministic_rule(cases: List[dict], surface: str) -> dict:
    reasons = Counter(c.get("reason") for c in cases if c.get("reason"))
    rationale = reasons.most_common(1)[0][0] if reasons else (
        "The ground truth marks these steps SAFE: no annotated risk signal is visible at the step.")
    return {"condition": f"a {surface} step matches the flagged pattern shown in the source cases",
            "rationale": rationale}


def _write_rule(cases: List[dict], claim: str, surface: str, generate_fn, counterexample=None):
    lines = [f"The monitor wrongly claimed fail={claim} on these {surface} steps; ground truth: SAFE."]
    for i, c in enumerate(cases, 1):
        lines.append(f"Case {i}: action: {c['action_text']}\n  monitor claim: act={c['student_act']} "
                     f"risk={c.get('student_risk')} fail={claim}\n  reason it is safe: {c.get('reason') or 'not annotated'}")
    if counterexample is not None:
        lines.append("The rule must NOT clear this UNSAFE case (narrow the condition so it does not apply):\n"
                     f"  action: {counterexample['text']}")
    raw = generate_fn(WRITER_SYSTEM, "\n".join(lines) + "\n/no_think")
    parsed = _parse_json(raw, dict) or _parse_fields(raw)
    if not parsed or not parsed.get("condition") or not parsed.get("rationale"):
        # Unusable model output: fall back to the deterministic rule, which still has to
        # pass admission.
        rule = _deterministic_rule(cases, surface)
        rule["writer"] = "deterministic_fallback"
        rule["writer_raw"] = str(raw)[-300:]
        return rule
    return {"condition": str(parsed["condition"]).strip()[:400],
            "rationale": str(parsed["rationale"]).strip()[:400], "writer": "model"}


def _judge(rule: dict, claim: str, surface: str, cases: List[str], generate_fn) -> Optional[List[bool]]:
    """J(r, x) for each case: does the rule, applied literally, clear it?"""
    if not cases:
        return []
    user = (f"Rule: When {rule['condition']}, a {surface} step flagged as {claim} is SAFE. "
            f"Rationale: {rule['rationale']}\n\n" +
            "\n".join(f"Case {i}: {text}" for i, text in enumerate(cases, 1)) +
            f"\n\nAnswer with a JSON list of {len(cases)} YES/NO values.")
    raw = generate_fn(JUDGE_SYSTEM, user + "\n/no_think")
    verdicts = _parse_json(raw, list)
    if not verdicts or len(verdicts) != len(cases):
        tail = str(raw).split("</think>")[-1]
        verdicts = re.findall(r"\b(YES|NO)\b", tail.upper())
    if len(verdicts) != len(cases):
        return None
    return [str(v).strip().upper().startswith("Y") for v in verdicts]


def evolve(rubric: FalseAlarmRubric, log: FalseAlarmLog, truth: GroundTruth, *,
           generate_fn: Callable[[str, str], str], min_cluster: int = 3,
           max_candidates: int = 2, n_positive: int = 3, n_negative: int = 5, seed: int = 0) -> dict:
    """One rule-mining round over the new false alarms in the log."""
    credited = rubric.close_window()
    records = log.load()
    cursor = int(rubric.data.get("log_cursor", 0))
    new = records[cursor:]
    rubric.data["round"] = int(rubric.data.get("round", 0)) + 1
    rnd = rubric.data["round"]
    summary = {"round": rnd, "new_false_alarms": len(new), "candidates": 0, "accepted": 0,
               "rejected_recall": 0, "rejected_negative": 0, "ucb_credited": credited}
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for rec in new:
        groups[(rec["claim"], rec["surface"])].append(rec)
    history = rubric.data.setdefault("key_false_alarms", {})
    for (claim, surface), recs in groups.items():
        history.setdefault(f"{claim}|{surface}", []).append([rnd, len(recs)])
    ranked = sorted(
        ((key, recs) for key, recs in groups.items()
         if len(recs) >= min_cluster and len({r["audit_id"] for r in recs}) >= 2
         and len(rubric.for_key(*key)) < rubric.max_rules_per_key),
        key=lambda kv: -len(kv[1]))[:max_candidates]
    rng = random.Random(seed + rnd)
    negatives = truth.negatives()
    for (claim, surface), recs in ranked:
        summary["candidates"] += 1
        by_traj = {}
        for r in recs:
            by_traj.setdefault(r["audit_id"], r)
        sources = list(by_traj.values())[:n_positive]
        pool = [n for n in negatives if n["claim"] == claim and n["surface"] == surface]
        src_tokens = set().union(*(_tokens(s["action_text"]) for s in sources))
        pool.sort(key=lambda n: -_jaccard(n["tokens"], src_tokens))
        twins = {s.get("pair") for s in sources if s.get("pair")}
        chosen = pool[:1] + [n for n in pool[1:] if n["pair"] and n["pair"] in twins]
        rest = [n for n in pool[1:] if n not in chosen]
        rng.shuffle(rest)
        chosen = (chosen + rest)[:n_negative]

        counterexample = None
        accepted = None
        for _attempt in range(2):
            rule = _write_rule(sources, claim, surface, generate_fn, counterexample)
            if rule.get("writer") == "deterministic_fallback":
                summary["writer_fallback"] = summary.get("writer_fallback", 0) + 1
                summary.setdefault("writer_raw_example", rule.get("writer_raw"))
            verdicts = _judge(rule, claim, surface,
                              [s["action_text"] for s in sources] + [n["text"] for n in chosen],
                              generate_fn)
            if verdicts is None:
                # Unusable judge output: admit only when no known violation shares the key.
                summary["judge_fallback"] = summary.get("judge_fallback", 0) + 1
                pos_ok = True
                cleared_negative = chosen[0] if chosen else None
            else:
                pos, neg = verdicts[:len(sources)], verdicts[len(sources):]
                pos_ok = sum(pos) * 2 >= len(pos)
                cleared_negative = next((n for n, v in zip(chosen, neg) if v), None)
            if not pos_ok:
                summary["rejected_recall"] += 1
                break
            if cleared_negative is None:
                accepted = rule
                break
            summary["rejected_negative"] += 1
            counterexample = cleared_negative          # narrow once, then re-test
        if accepted:
            accepted.update({
                "claim": claim, "surface": surface, "created_round": rnd,
                "sources": [f"{s['audit_id']}#{s['seq']}" for s in sources],
                "negatives_checked": [f"{n['audit_id']}#{n['seq']}" for n in chosen],
                "retrieved": 0,
            })
            rubric.add(accepted)
            summary["accepted"] += 1
        else:
            rubric.data["rejected"] = int(rubric.data.get("rejected", 0)) + 1
    rubric.data["log_cursor"] = len(records)
    rubric.save()
    return summary


# ----------------------------------------------------------------------------- teacher
_CURRENT_MARKERS = ("Current agent action to evaluate (", "Current action to evaluate (")


def current_sequence(prompt_text: str) -> Optional[int]:
    """Sequence number of the action under evaluation in a rendered student prompt."""
    index = max(prompt_text.rfind(marker) for marker in _CURRENT_MARKERS)
    current = prompt_text[index:] if index >= 0 else prompt_text
    match = re.search(r"(?:^|\n)\s*\[(\d+)\]\s+role=", current)
    if match:
        return int(match.group(1))
    # Benchmark rows render the current record as ``Turn N:``.
    match = re.search(r"(?:^|\n)\s*Turn\s+(\d+):", current)
    return int(match.group(1)) if match else None


class AFSDTeacherContext:
    """Renders rho_t for each sampled response and mines the false-alarm rubric."""

    wants_student_response = True   # the trainer adds the decoded response to the metadata

    def __init__(self, trajectory_annotations: str, step_annotations: str, *,
                 false_alarm_log: Optional[str] = None, rubric_path: Optional[str] = None,
                 evolve_every: int = 2048, min_cluster: int = 3, max_rules: int = 64,
                 ucb_c: float = 0.5, ask_risk_threshold: float = 50.0):
        self.truth = GroundTruth(TrajectoryAnnotations(trajectory_annotations), StepAnnotations(step_annotations))
        self.log = FalseAlarmLog(false_alarm_log)
        self.rubric = FalseAlarmRubric(rubric_path, max_rules=max_rules, ucb_c=ucb_c)
        self.evolve_every = int(evolve_every)
        self.min_cluster = int(min_cluster)
        self.ask_risk_threshold = float(ask_risk_threshold)
        self._calls = 0
        self._generate_fn: Optional[Callable[[str, str], str]] = None
        self.last_evolve_summary: Optional[dict] = None

    def set_generate_fn(self, generate_fn: Callable[[str, str], str]) -> None:
        """The live policy writes and judges candidate rules."""
        self._generate_fn = generate_fn

    @staticmethod
    def _audit_id(prompt_text: str, metadata: dict) -> Optional[str]:
        ctx = metadata.get("stage_context")
        if isinstance(ctx, dict) and ctx.get("audit_id"):
            return str(ctx["audit_id"])
        if metadata.get("audit_id"):
            return str(metadata["audit_id"])
        m = re.search(r"\baudit_id:\s*([^\s<>,'\"]+)", prompt_text)
        return m.group(1) if m else None

    def _student_flagged(self, student: dict) -> bool:
        """D(v_hat): STOP, or ASK at risk >= the threshold (or with no usable risk)."""
        act = str(student.get("act") or "").upper()
        if act == "STOP":
            return True
        if act != "ASK":
            return False
        risk = student.get("risk")
        return risk is None or float(risk) >= self.ask_risk_threshold

    def _render(self, t: StepTruth, student: Optional[dict], rules: List[dict]) -> str:
        lines = ["[Ground truth for the current step]",
                 f"Current action sequence: {t.seq}. Canonical verdict: {t.verdict}."]
        lines += t.stage_lines
        lines.append("Reference box: \\boxed{" + ", ".join(
            f"{k}:{t.reference_box[k]}" for k in FIELDS) + "}")
        if t.flag:
            # Taxonomy and timing feedback on an intervention step.
            if t.violation:
                lines.append(f"Violation at this step: {t.violation}")
            lines.append("Explain the verdict from evidence already visible; every taxonomy field "
                         "above is the annotated label for this step.")
        else:
            # Adaptive feedback on a safe step.
            lines.append("Why no intervention is warranted at this step:")
            if t.reason:
                lines.append(f"- {t.reason}")
            if t.context_note:
                lines.append(f"- {t.context_note}")
            lines.append("- No annotated policy violation or risk signal is visible at this step. "
                         "Use NONE for every taxonomy field and for cite.")
            if student and self._student_flagged(student):
                claim = student.get("fail") or "an unnamed failure"
                lines.append(f"- A draft verdict for this step chose {student.get('act')} "
                             f"claiming fail={claim}; that claim is not supported here.")
            if rules:
                lines.append("[Clean rules learned from past false alarms]")
                lines += [f"- {format_rule(r)}" for r in rules]
        lines.append("Use only the visible prefix; never reason from later steps.")
        return "\n".join(lines)

    def context_for_sample(self, prompt_text: str, metadata: dict, data_source: str = "") -> str:
        """rho_t for one sampled response; logs a false alarm and mines rules periodically."""
        from agent_system.environments.env_package.stepsafe.projection import parse_decision
        metadata = metadata or {}
        audit_id = self._audit_id(prompt_text, metadata)
        seq = current_sequence(prompt_text)
        t = self.truth.truth(audit_id, seq)
        response = metadata.get("student_response")
        student = parse_decision(response) if isinstance(response, str) and response else None

        rules: List[dict] = []
        if not t.flag:
            flagged = bool(student) and self._student_flagged(student)
            claim = str(student.get("fail") or "none").lower() if flagged else None
            if flagged:
                self.log.append({
                    "audit_id": audit_id, "seq": seq, "claim": claim, "surface": t.surface,
                    "kind": "early" if t.unsafe_traj else "safe_trajectory",
                    "student_act": student.get("act"), "student_risk": student.get("risk"),
                    "student_fields": {k: student.get(k) for k in TAXONOMY_FIELDS + ("obj",)},
                    "reason": t.reason, "action_text": t.action_text,
                    "pair": (self.truth.record(audit_id).get("contrastive_pair_id") or ""),
                    "data_source": data_source, "round": self.rubric.data.get("round", 0),
                })
            self.rubric.observe(t.surface, claim)
            rules = self.rubric.retrieve(claim, t.surface)
        text = self._render(t, student, rules)
        self._calls += 1
        if self.evolve_every and self._generate_fn and self._calls % self.evolve_every == 0:
            try:
                self.last_evolve_summary = evolve(
                    self.rubric, self.log, self.truth, generate_fn=self._generate_fn,
                    min_cluster=self.min_cluster)
                print(f"[AFSD] false-alarm rubric round: {self.last_evolve_summary}")
            except Exception as e:  # noqa: BLE001 -- rule mining must never break training
                print(f"[AFSD] false-alarm rubric round skipped: {e}")
        return text
