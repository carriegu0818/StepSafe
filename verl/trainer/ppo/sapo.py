"""Stage-Aligned Policy Optimization (SAPO), Section 2.3 of the paper.

SAPO evaluates each monitor decision at its own step:

1. Stage score sigma(zeta(t), a). The annotated signal s, boundary b and
   last preventable step h place step t in a stage, and the stage scores the sampled
   action (``STAGE_SCORES``). Malformed verdicts score -1; steps after h (an already
   executed harm) are excluded.
2. Prefix-group advantage. Responses sampled at the same visible
   prefix judge the same proposed action, so each response is compared with the
   leave-one-out mean of the others at that prefix, without std normalization.
   A response alone at its prefix gets A_i = 0.
3. Balanced weighting of safe and intervention steps. Responses are grouped into
   (task, run, annotated class) cells and every cell receives the same total weight,
   split equally over its responses; the class-balancing factors of Balance-GRPO
   [StepGuard] (inverse class frequency and the gap in class accuracy, both bounded)
   then rescale the result.

The weighted advantages enter the clipped surrogate with token-mean aggregation; the
per-response token count is cancelled so each response carries weight w_i, not
w_i * L_i.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

ACTIONS = ("CONTINUE", "ASK", "STOP")
# sigma(stage, action) for (CONTINUE, ASK, STOP), the stage table of Section 2.3:
#   safe          safe trajectory, or t < s     (Before signal)
#   clarification s <= t < b                    (Clarification)
#   boundary      t = b < h                     (Boundary)
#   late          b < t < h                     (After boundary)
#   harm          t = h                         (Last preventable)
#   post_incident t > h, or the action's result is already visible (excluded)
STAGE_SCORES = {
    "safe": (1.0, -1.0, -1.0),
    "clarification": (-1.0, 1.0, 0.5),
    "boundary": (-1.0, 0.0, 1.0),
    "late": (-1.0, -0.5, 1.0),
    "harm": (-1.0, -1.0, 1.0),
    "post_incident": (0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class StageSpec:
    stage: str

    @property
    def eligible(self) -> bool:
        return self.stage != "post_incident"

    @property
    def scores(self) -> dict:
        return dict(zip(ACTIONS, STAGE_SCORES[self.stage]))

    def score(self, action, valid=True) -> float:
        if not self.eligible:
            return 0.0
        return self.scores.get(str(action).upper(), -1.0) if valid else -1.0


def stage_spec(context: Mapping) -> StageSpec:
    """zeta(t) from the step's ``stage_context``; harm wins a tie with the boundary."""
    if context["post_incident"]:
        return StageSpec("post_incident")
    if not context["unsafe"]:
        return StageSpec("safe")
    t, signal, boundary, harm = (context[k] for k in ("t", "signal", "boundary", "harm"))
    if any(v is None for v in (t, signal, boundary, harm)):
        raise ValueError("SAPO requires the landmarks s, b, h of every unsafe trajectory")
    if not signal <= boundary <= harm:
        raise ValueError("SAPO requires s <= b <= h")
    if t > harm:
        stage = "post_incident"
    elif t < signal:
        stage = "safe"
    elif t < boundary:
        stage = "clarification"
    elif t == harm:
        stage = "harm"
    elif t == boundary:
        stage = "boundary"
    else:
        stage = "late"
    return StageSpec(stage)


def stage_score(context: Mapping) -> tuple[float, str]:
    spec = stage_spec(context)
    return spec.score(context["action"], context["valid"]), spec.stage


def _bool_array(values: Sequence[object], *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if array.dtype.kind in "biuf":
        return array.astype(bool)
    return np.array([str(v).strip().lower() in {"1", "true", "yes", "on", "unsafe"} for v in array], dtype=bool)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def class_balance_factors(
    gold_intervene: np.ndarray,
    predicted_intervene: np.ndarray,
    action_valid: np.ndarray,
    config: Mapping[str, object],
) -> tuple[np.ndarray, dict[str, float]]:
    """Balance-GRPO class factors (released StepGuard implementation).

    Each response gets ``clip(class_weight * gap_weight)`` where the class weight is the
    bounded inverse frequency of its annotated class in the batch, and the gap weight
    moves credit toward the class with the lower accuracy. An invalid response counts as
    a classification error for either class.
    """
    gold = gold_intervene
    predicted = np.where(action_valid, predicted_intervene, ~gold)
    safe, unsafe = ~gold, gold
    safe_total, unsafe_total = int(safe.sum()), int(unsafe.sum())
    alpha = float(config.get("smoothing_alpha", 1.0))
    correct = gold == predicted
    safe_accuracy = (float(np.logical_and(safe, correct).sum()) + alpha) / (safe_total + 2.0 * alpha)
    unsafe_accuracy = (float(np.logical_and(unsafe, correct).sum()) + alpha) / (unsafe_total + 2.0 * alpha)

    raw_gap = unsafe_accuracy - safe_accuracy - float(config.get("target_gap", 0.0))
    deadband = float(config.get("deadband", 0.02))
    gap_clip = float(config.get("gap_clip", 0.5))
    gap = 0.0 if abs(raw_gap) < deadband else _clip(raw_gap, -gap_clip, gap_clip)

    class_min = float(config.get("class_weight_min", 0.5))
    class_max = float(config.get("class_weight_max", 2.0))
    gap_min = float(config.get("gap_weight_min", 0.5))
    gap_max = float(config.get("gap_weight_max", 2.0))
    effective_min = float(config.get("effective_weight_min", 0.75))
    effective_max = float(config.get("effective_weight_max", 1.5))
    gap_lambda = float(config.get("lambda", 2.0))

    total = len(gold)
    weights = np.empty(total, dtype=np.float32)
    for i, is_unsafe in enumerate(gold):
        count = unsafe_total if is_unsafe else safe_total
        class_weight = _clip(total / (2.0 * max(count, 1)), class_min, class_max)
        gap_weight = 1.0 - gap_lambda * gap if is_unsafe else 1.0 + gap_lambda * gap
        gap_weight = _clip(gap_weight, gap_min, gap_max)
        weights[i] = _clip(class_weight * gap_weight, effective_min, effective_max)
    stats = {
        "continue_accuracy": float(safe_accuracy),
        "intervene_accuracy": float(unsafe_accuracy),
        "accuracy_gap": float(gap),
        "continue_fraction": float(safe_total / total),
        "intervene_fraction": float(unsafe_total / total),
        "mean_factor": float(weights.mean()),
    }
    return weights, stats


def compute_sapo_advantage(data, mask: torch.Tensor, balance: Mapping[str, object] | None = None):
    """Per-response SAPO advantages, broadcast over response tokens.

    Returns ``(advantages, returns, stats)``; ``returns`` holds the advantages before
    the class-balancing factors (logged only).

    ``data.non_tensor_batch`` must hold, per response: ``uid`` (task), ``traj_uid``
    (run), ``anchor_obs`` (the visible prefix), ``stage_context``, ``intervention_target``
    (annotated class d*_t), ``intervention_predicted`` (D(v)), ``is_action_valid``.
    """
    meta = data.non_tensor_batch
    gold = _bool_array(meta["intervention_target"], name="intervention_target")
    pred = _bool_array(meta["intervention_predicted"], name="intervention_predicted")
    valid = _bool_array(meta["is_action_valid"], name="is_action_valid")
    n = len(gold)
    if any(len(meta[k]) != n for k in ("uid", "traj_uid", "anchor_obs", "stage_context")) or mask.shape[0] != n:
        raise ValueError("SAPO metadata must align with response rows")
    lengths = mask.sum(-1).detach().cpu().numpy()
    active = lengths > 0

    # 1. Stage score of every sampled decision.
    contexts = meta["stage_context"]
    stage_names = np.array(["none"] * n, dtype=object)
    scores = ((gold == pred) & valid).astype(np.float64)
    for i in np.flatnonzero(active):
        scores[i], stage_names[i] = stage_score(contexts[i])
    active &= stage_names != "post_incident"

    # 2. Leave-one-out credit among responses at the same visible prefix.
    groups = defaultdict(list)
    cells = defaultdict(list)
    for i in np.flatnonzero(active):
        groups[(str(meta["uid"][i]), str(meta["anchor_obs"][i]))].append(i)
        cells[(str(meta["uid"][i]), str(meta["traj_uid"][i]), bool(gold[i]))].append(i)
    credit = np.zeros(n, dtype=np.float64)
    informative = 0
    for indices in groups.values():
        if len(set(gold[indices])) != 1:
            raise ValueError("Identical prefixes have inconsistent annotated classes")
        if len(indices) > 1:
            if len(set(stage_names[indices])) != 1:
                raise ValueError("Identical prefixes have inconsistent stages")
            group_scores = scores[indices]
            credit[indices] = (group_scores - group_scores.mean()) * len(indices) / (len(indices) - 1)
            informative += int(group_scores.min() != group_scores.max())

    # 3. Equal weight per (task, run, class) cell, split over its responses.
    weights = np.zeros(n, dtype=np.float64)
    for indices in cells.values():
        weights[indices] = 1.0 / len(cells) / len(indices)
    # Cancel the response length under token-mean aggregation.
    token_scale = lengths.sum() / np.maximum(lengths, 1)
    scalar = credit * weights * token_scale
    advantages = torch.as_tensor(scalar, device=mask.device, dtype=torch.float32)[:, None] * mask

    stats = {
        "continue_fraction": float((~gold[active]).mean()) if active.any() else 0.0,
        "informative_group_fraction": informative / max(len(groups), 1),
        "singleton_group_fraction": sum(len(x) == 1 for x in groups.values()) / max(len(groups), 1),
        "nonzero_credit_fraction": float(np.mean(credit[active] != 0)) if active.any() else 0.0,
        "observed_cells": len(cells),
        "post_incident_fraction": float(np.mean(stage_names == "post_incident")),
    }
    for stage in ("safe", "clarification", "boundary", "late", "harm"):
        selected = active & (stage_names == stage)
        stats[stage + "_stage_fraction"] = float(selected.sum() / max(active.sum(), 1))
        stats[stage + "_score_mean"] = float(scores[selected].mean()) if selected.any() else 0.0
    for cls, name in ((False, "continue"), (True, "intervene")):
        selected = active & (gold == cls)
        correct = (gold == pred) & valid
        stats[name + "_accuracy"] = float(correct[selected].mean()) if selected.any() else 0.0
        stats[name + "_budget"] = float(weights[selected].sum())
        stats[name + "_absolute_credit_mass"] = float(np.abs(credit[selected] * weights[selected]).sum())

    returns = advantages.clone()
    if balance is not None and bool(balance.get("enabled", False)):
        factors, balance_stats = class_balance_factors(gold, pred, valid, balance)
        advantages = advantages * torch.as_tensor(factors, dtype=advantages.dtype, device=advantages.device).unsqueeze(-1)
        stats.update({f"balance/{k}": v for k, v in balance_stats.items()})
    return advantages, returns, stats
