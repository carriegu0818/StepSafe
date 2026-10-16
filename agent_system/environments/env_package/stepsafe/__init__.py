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

"""StepSafe replay environment.

Submodules are imported lazily so the pure-logic parts (``labels``, ``reward``,
``projection``) can be used without the training stack.
"""

from __future__ import annotations

__all__ = ["build_monitor_replay_envs", "StepSafeEnvironmentManager", "stepsafe_projection"]


def __getattr__(name: str):  # PEP 562 lazy attribute loading
    if name == "build_monitor_replay_envs":
        from .envs import build_monitor_replay_envs
        return build_monitor_replay_envs
    if name == "StepSafeEnvironmentManager":
        from .manager import StepSafeEnvironmentManager
        return StepSafeEnvironmentManager
    if name == "stepsafe_projection":
        from .projection import stepsafe_projection
        return stepsafe_projection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
