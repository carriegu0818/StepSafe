"""SAPO: stage scores, prefix-group credit, cell weights and class balancing."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo.sapo import STAGE_SCORES, compute_sapo_advantage, stage_score, stage_spec


def ctx(t, action, unsafe=True, s=3, b=5, h=6, valid=True, post_incident=False):
    return {"audit_id": "x", "unsafe": unsafe, "t": t, "signal": s, "boundary": b, "harm": h,
            "post_incident": post_incident, "action": action, "valid": valid}


def test_stage_table_matches_paper():
    # (CONTINUE, ASK, STOP) per stage, Section 2.3.
    assert STAGE_SCORES["safe"] == (1.0, -1.0, -1.0)
    assert STAGE_SCORES["clarification"] == (-1.0, 1.0, 0.5)
    assert STAGE_SCORES["boundary"] == (-1.0, 0.0, 1.0)
    assert STAGE_SCORES["late"] == (-1.0, -0.5, 1.0)
    assert STAGE_SCORES["harm"] == (-1.0, -1.0, 1.0)


@pytest.mark.parametrize("t,stage", [(2, "safe"), (3, "clarification"), (4, "clarification"),
                                     (5, "boundary"), (6, "harm"), (7, "post_incident")])
def test_stages_of_unsafe_trajectory(t, stage):
    assert stage_spec(ctx(t, "STOP")).stage == stage


def test_late_stage_and_harm_wins_tie():
    assert stage_spec(ctx(5, "STOP", s=2, b=4, h=6)).stage == "late"
    assert stage_spec(ctx(4, "STOP", s=2, b=4, h=4)).stage == "harm"
    assert stage_spec(ctx(1, "STOP", unsafe=False)).stage == "safe"


def test_invalid_verdict_scores_minus_one():
    assert stage_score(ctx(3, "ASK", valid=False)) == (-1.0, "clarification")
    assert stage_score(ctx(3, "ASK")) == (1.0, "clarification")


def batch(rows, lengths):
    keys = ("uid", "traj_uid", "anchor_obs", "stage_context", "intervention_target", "intervention_predicted", "is_action_valid")
    nt = {k: np.array([r[i] for r in rows], dtype=object) for i, k in enumerate(keys)}
    mask = torch.zeros(len(rows), max(lengths))
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
    return SimpleNamespace(non_tensor_batch=nt, batch={}), mask


def test_leave_one_out_credit_and_cell_weights():
    # One task, three runs at the same clarification-stage prefix: ASK, STOP, CONTINUE.
    rows = [("u", f"r{k}", "prefix", ctx(3, a), True, a != "CONTINUE", True)
            for k, a in enumerate(["ASK", "STOP", "CONTINUE"])]
    data, mask = batch(rows, [4, 4, 4])
    adv, ret, stats = compute_sapo_advantage(data, mask, balance=None)
    scores = np.array([1.0, 0.5, -1.0])
    loo = (scores - scores.mean()) * 3 / 2
    # Three (task, run, class) cells of one response each: weight 1/3; token scale 12/4 = 3.
    expected = loo * (1 / 3) * 3
    assert torch.allclose(adv[:, 0], torch.tensor(expected, dtype=torch.float32))
    assert torch.equal(adv, ret)
    assert stats["informative_group_fraction"] == 1.0


def test_singleton_prefix_and_post_incident_get_no_credit():
    rows = [("u", "r0", "p0", ctx(3, "ASK"), True, True, True),
            ("u", "r1", "p1", ctx(7, "STOP"), True, True, True)]
    data, mask = batch(rows, [5, 5])
    adv, _, stats = compute_sapo_advantage(data, mask)
    assert torch.count_nonzero(adv) == 0
    assert stats["post_incident_fraction"] == 0.5


def test_balance_factors_rescale_advantages():
    rows = [("u", f"r{k}", "p", ctx(1, a, unsafe=False), False, a != "CONTINUE", True)
            for k, a in enumerate(["CONTINUE", "STOP"])]
    rows += [("v", f"q{k}", "p2", ctx(3, a), True, a != "CONTINUE", True) for k, a in enumerate(["ASK", "CONTINUE"])]
    data, mask = batch(rows, [3, 3, 3, 3])
    plain, _, _ = compute_sapo_advantage(data, mask)
    balanced, _, stats = compute_sapo_advantage(data, mask, balance={"enabled": True})
    assert stats["balance/continue_fraction"] == 0.5
    assert torch.allclose(balanced, plain)  # equal classes and equal accuracy: factors are 1
