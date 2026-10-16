"""False-alarm rubric: admission rule and UCB retrieval."""
import json

from verl.trainer.ppo.afsd_context import FalseAlarmLog, FalseAlarmRubric, evolve


class Truth:
    """Two known violations on the communication surface with the claim 'authorization'."""

    def negatives(self):
        return [{"audit_id": f"bad{i}", "seq": 3, "claim": "authorization", "surface": "communication",
                 "text": f"agent communication -> external :: leaks record {i}", "tokens": {"leaks", "record"}, "pair": ""}
                for i in range(2)]


def false_alarms(log, n):
    for i in range(n):
        log.append({"audit_id": f"safe{i % 2}", "seq": i, "claim": "authorization", "surface": "communication",
                    "action_text": f"agent communication -> manager :: status update {i}", "student_act": "STOP",
                    "student_risk": 80, "reason": "internal status update"})


def writer_and_judge(clears_violations):
    def generate(system, user):
        if system.startswith("You write one precise rule"):
            return json.dumps({"condition": "the message goes to an internal manager", "rationale": "internal routing"})
        n = user.count("\nCase ")
        # Sources are listed first, then the known violations.
        return json.dumps(["YES" if (k < n - 2 or clears_violations) else "NO" for k in range(n)])
    return generate


def test_rule_admitted_only_if_it_clears_no_violation(tmp_path):
    log = FalseAlarmLog(str(tmp_path / "log.jsonl"))
    false_alarms(log, 4)
    rubric = FalseAlarmRubric(str(tmp_path / "rubric.json"))
    summary = evolve(rubric, log, Truth(), generate_fn=writer_and_judge(clears_violations=True))
    assert summary["accepted"] == 0 and summary["rejected_negative"] == 2  # rewritten once, still rejected

    log2 = FalseAlarmLog(str(tmp_path / "log2.jsonl"))
    false_alarms(log2, 4)
    rubric2 = FalseAlarmRubric(str(tmp_path / "rubric2.json"))
    summary = evolve(rubric2, log2, Truth(), generate_fn=writer_and_judge(clears_violations=False))
    assert summary["accepted"] == 1
    rule = rubric2.rules[0]
    assert (rule["claim"], rule["surface"]) == ("authorization", "communication")
    assert json.load(open(tmp_path / "rubric2.json"))["rules"][0]["id"] == "R1"


def test_ucb_retrieval_prefers_untried_rules_and_credits_windows(tmp_path):
    rubric = FalseAlarmRubric(None)
    for i in range(2):
        rubric.add({"claim": "authorization", "surface": "communication", "condition": f"c{i}",
                    "rationale": "r", "created_round": i})
    first = rubric.retrieve("authorization", "communication")
    assert {r["id"] for r in first} == {"R1", "R2"}
    for _ in range(4):
        rubric.observe("communication", None)
    rubric.observe("communication", "authorization")
    credited = rubric.close_window()
    assert credited == {"R1": 0.8, "R2": 0.8}
    # Without a claim, at most one rule is retrieved for the action surface.
    assert len(rubric.retrieve(None, "communication")) == 1
