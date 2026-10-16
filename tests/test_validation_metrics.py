"""Validation metrics (val/*, paper/*) and the W&B metric filter."""
from types import SimpleNamespace

import pytest

from verl.trainer.ppo.aspo_trainer import PAPER_TRAINING_METRICS, ASPORayTrainer
from verl.trainer.ppo.monitor_metrics import paper_table_metrics, validation_metrics


def record(unsafe, stop=None, high_risk_ask=None, blocked=None):
    return {"unsafe": unsafe, "first_stop_sequence": stop, "first_high_risk_ask_sequence": high_risk_ask,
            "supports_timing": blocked is not None, "blocked_before_harm": blocked}


def test_validation_metrics():
    records = [
        record(True, stop=3, blocked=True),            # caught in time
        record(True, high_risk_ask=4, blocked=False),  # flagged, harm not prevented
        record(True),                                  # missed, no timing
        record(False),                                 # clean pass
        record(False, stop=2),                         # false alarm
    ]
    m = validation_metrics(records)
    assert set(m) == {"val/j", "val/asr", "val/utility"}
    assert m["val/utility"] == pytest.approx(0.5)
    assert m["val/j"] == pytest.approx(2 / 3 + 0.5 - 1)
    assert m["val/asr"] == pytest.approx(0.5)


def episode(source, unsafe, acts):
    return {"audit_id": f"{source}-{unsafe}-{len(acts)}", "data_source": source, "gt_unsafe": unsafe,
            "decisions": [{"cursor": i, "act": a, "risk": 80} for i, a in enumerate(acts)],
            "monitor_record": {"supports_timing": False}}


def test_paper_metrics_keys():
    episodes = [episode("assebench_safety", True, ["CONTINUE", "STOP"]),
                episode("assebench_safety", False, ["CONTINUE", "CONTINUE"]),
                episode("atbench_1000", True, ["CONTINUE"]),
                episode("atbench_1000", False, ["ASK"])]
    m = paper_table_metrics(episodes, agentharm_labels={})
    assert {k.rsplit("/", 1)[1] for k in m} <= {"acc", "mf1", "util", "j", "asr", "bir", "ble", "stop"}
    assert {k.split("/")[1] for k in m} == {"ASSE-Safety", "ATBench", "Overall"}
    assert m["paper/ASSE-Safety/acc"] == pytest.approx(100.0)
    assert m["paper/ATBench/acc"] == pytest.approx(0.0)
    assert m["paper/Overall/j"] == pytest.approx(0.0)


def test_paper_metrics_ignore_non_benchmark_rows():
    benchmark = [episode("assebench_rjudge", True, ["STOP"]), episode("assebench_rjudge", False, ["CONTINUE"])]
    other = [episode("agentdog", False, ["STOP"])]
    assert paper_table_metrics(benchmark + other, {}) == paper_table_metrics(benchmark, {})


@pytest.mark.parametrize("log_all", [False, True])
def test_wandb_filter(log_all):
    trainer = SimpleNamespace(config=SimpleNamespace(trainer={"log_all_metrics": log_all}))
    metrics = {key: 0.0 for key in PAPER_TRAINING_METRICS}
    metrics.update({"val/j": 0.5, "paper/Overall/j": 50.0, "actor/pg_clipfrac": 0.1,
                    "afsd/gate_mean": 0.3, "timing_s/gen": 1.0, "critic/score/mean": 0.2})
    logged = ASPORayTrainer._loggable(trainer, metrics)
    if log_all:
        assert logged == metrics
    else:
        assert set(logged) == set(PAPER_TRAINING_METRICS) | {"val/j", "paper/Overall/j"}
