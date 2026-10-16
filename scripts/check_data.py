"""Check a StepSafe data directory before training (CPU, a few seconds).

    python scripts/check_data.py            # data/ of this repository, after scripts/build_data.py

Checks that
  * every RL replay task carries the monitor protocol of agent_system/environments/prompts/stepsafe.py
    as its system prompt, and that the step verdicts embedded in each task agree with the step
    annotations the AFSD teacher reads (rl/step_annotations.jsonl);
  * every RL task has a trajectory annotation (rl/trajectory_annotations.jsonl);
  * the release sizes match the paper: 3,184 RL tasks with 22,000 annotated steps, 500
    validation episodes (100 each of ASSE-Safety, ASSE-Security, R-Judge, AgentHarm and ATBench),
    4,033 evaluation trajectories (ASSE-Safety 1,476, ASSE-Security 817, R-Judge 564, AgentHarm 176,
    ATBench 1,000);
  * validation and evaluation contain only these five benchmarks.
"""

import argparse
import collections
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agent_system.environments.prompts.stepsafe import FIELDS, render_protocol  # noqa: E402
from verl.trainer.ppo.afsd_context import StepAnnotations  # noqa: E402

EVAL_SOURCES = {"assebench_safety": 1476, "assebench_security": 817, "assebench_rjudge": 564,
                "assebench_agentharm": 176, "atbench_1000": 1000}
VALIDATION = {"assebench.parquet": {"assebench_safety": 100, "assebench_security": 100,
                                    "assebench_rjudge": 100, "assebench_agentharm": 100},
              "atbench.parquet": {"atbench_1000": 100}}


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def check_rl(data, expect_rows, expect_steps):
    rows = pq.read_table(os.path.join(data, "rl", "train.parquet")).to_pylist()
    steps = StepAnnotations(os.path.join(data, "rl", "step_annotations.jsonl"))
    trajectories = set()
    with open(os.path.join(data, "rl", "trajectory_annotations.jsonl")) as f:
        for line in f:
            if line.strip():
                trajectories.add(json.loads(line)["audit_id"])
    protocol = render_protocol()
    n_steps = n_intervene = 0
    for i, row in enumerate(rows):
        prompt = _json(row["prompt"])
        if len(prompt) != 2 or prompt[0].get("role") != "system" or prompt[0].get("content") != protocol:
            raise SystemExit(f"rl/train.parquet row {i}: system prompt differs from the monitor protocol")
        env = _json(row["env_kwargs"])
        if env["audit_id"] not in trajectories:
            raise SystemExit(f"rl/train.parquet row {i}: no trajectory annotation for {env['audit_id']}")
        for item in _json(env["v2_foresight_step_boxes"]):
            box = item["box"]
            if tuple(box) != FIELDS:
                raise SystemExit(f"rl/train.parquet row {i}: verdict fields out of order at step {item.get('sequence_num')}")
            reference = steps.records.get((env["audit_id"], item["sequence_num"]))
            if reference is None or reference["box"] != box:
                raise SystemExit(f"rl/train.parquet row {i}: step {item['sequence_num']} disagrees with rl/step_annotations.jsonl")
            n_steps += 1
            n_intervene += box["act"] != "CONTINUE"
    if len(rows) != expect_rows or n_steps != expect_steps:
        raise SystemExit(f"unexpected RL release size: tasks={len(rows)} annotated steps={n_steps}")
    print(f"[check] rl: {len(rows)} tasks, {n_steps} annotated steps ({n_intervene} ASK/STOP), "
          f"{len(steps.records)} step annotations, {len(trajectories)} trajectory annotations")


def check_validation(data):
    total = 0
    for name, expected in VALIDATION.items():
        path = os.path.join(data, "eval", "validation", name)
        rows = pq.read_table(path, columns=["data_source"]).to_pylist()
        counts = collections.Counter(r["data_source"] for r in rows)
        if dict(counts) != expected:
            raise SystemExit(f"eval/validation/{name}: unexpected sources {dict(counts)}")
        total += len(rows)
    print(f"[check] validation: {total} episodes")


def check_eval(data):
    rows = pq.read_table(os.path.join(data, "eval", "table1.parquet"), columns=["data_source"]).to_pylist()
    counts = collections.Counter(r["data_source"] for r in rows)
    if dict(counts) != EVAL_SOURCES:
        raise SystemExit(f"eval/table1.parquet: unexpected sources {dict(counts)}")
    with open(os.path.join(data, "eval", "agentharm_step_labels.jsonl")) as f:
        labels = [json.loads(line) for line in f if line.strip()]
    if len(labels) != EVAL_SOURCES["assebench_agentharm"]:
        raise SystemExit(f"eval/agentharm_step_labels.jsonl: {len(labels)} rows, expected 176")
    print(f"[check] eval: {len(rows)} trajectories {dict(counts)}; {len(labels)} AgentHarm harm-step labels")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
                        help="data directory (rl/, eval/) [data]")
    parser.add_argument("--expect-tasks", type=int, default=3184)
    parser.add_argument("--expect-steps", type=int, default=22000)
    args = parser.parse_args()
    check_rl(args.data, args.expect_tasks, args.expect_steps)
    check_validation(args.data)
    check_eval(args.data)
    print("[check] PASS")


if __name__ == "__main__":
    main()
