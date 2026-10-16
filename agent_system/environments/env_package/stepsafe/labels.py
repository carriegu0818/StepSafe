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

"""Intervention-window annotations of a trajectory and record rendering.

For an unsafe trajectory the annotation records three landmarks (Section 2.2):

    t_signal   (s): first step at which the risk is observable from the visible prefix;
    t_boundary (b): the step whose execution violates the policy;
    t_harm     (h): the last preventable step, with s <= b <= h.

A safe trajectory has none. ``context_risk`` marks a risky *input* (e.g. a harmful
user request) that is context, not an agent violation and not a stopping target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from multi_agent.schemas.actions import ObservableAction

TOOL_RESULT_EXCERPT_CHARS = 1200


@dataclass
class Timepoints:
    """Annotated landmarks of one trajectory."""

    unsafe: bool
    t_signal: int | None = None
    t_boundary: int | None = None
    t_harm: int | None = None
    channel: str | None = None
    risk_object: str | None = None
    cf_safe: str | None = None
    context_risk: bool = False
    t_context_risk: int | None = None
    context_risk_object: str | None = None
    # Rows that leave the channel unlabeled switch off channel/object matching.
    require_channel: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Timepoints":
        tp = Timepoints(**{k: d.get(k) for k in (
            "unsafe", "t_signal", "t_boundary", "t_harm", "channel",
            "risk_object", "cf_safe", "context_risk", "t_context_risk",
            "context_risk_object")})
        rc = d.get("require_channel")
        tp.require_channel = True if rc is None else bool(rc)
        tp.context_risk = bool(d.get("context_risk", False))
        return tp


def render_action(a: ObservableAction) -> str:
    """One trajectory record as the monitor sees it."""
    if a.surface == "tool_call":
        args = a.tool_args or {}
        arg_s = ", ".join(f"{k}={v}" for k, v in args.items() if v not in (None, ""))
        rendered = f"[{a.sequence_num}] role={a.agent_role} tool_call {a.tool_name}({arg_s})"
        result = str(a.tool_result or "").strip()
        if result:
            if len(result) > TOOL_RESULT_EXCERPT_CHARS:
                half = TOOL_RESULT_EXCERPT_CHARS // 2
                result = f"{result[:half]} ...[truncated]... {result[-half:]}"
            rendered += f" -> {result}"
        return rendered
    return f"[{a.sequence_num}] role={a.agent_role} message -> {a.target_role}: {a.content or ''}"


def render_action_verbatim(a: ObservableAction) -> str:
    """Benchmark rows (ASSEBench) whose records already carry the source's own
    rendered text in ``content``; shown as-is."""
    return f"Turn {a.sequence_num}:\n{a.content or ''}"
