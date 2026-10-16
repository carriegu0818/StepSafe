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

from functools import partial

from omegaconf import OmegaConf

from agent_system.environments.base import EnvironmentManagerBase  # noqa: F401  (re-exported)


def _validation_config(config):
    """Validation replays use ``config.env`` with the overrides of ``env.validation``."""
    env = OmegaConf.to_container(config.env, resolve=True)
    overrides = env.pop("validation", None) or {}
    val_env = {**env, **overrides}
    return OmegaConf.create({"env": val_env, "data": OmegaConf.to_container(config.data, resolve=True)})


def make_envs(config):
    """Training and validation replay environments for StepSafe."""
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    if config.env.env_name.lower() != "stepsafe":
        raise ValueError(f"Unsupported environment: {config.env.env_name}")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

    from agent_system.environments.env_package.stepsafe import (
        StepSafeEnvironmentManager,
        build_monitor_replay_envs,
        stepsafe_projection,
    )

    # G replays of each of the train_batch_size tasks.
    _envs = build_monitor_replay_envs(
        seed=config.env.seed, env_num=config.data.train_batch_size,
        group_n=group_n, is_train=True, env_config=config.env)
    val_config = _validation_config(config)
    _val_envs = build_monitor_replay_envs(
        seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
        group_n=1, is_train=False, env_config=val_config.env)
    projection_f = partial(stepsafe_projection)
    envs = StepSafeEnvironmentManager(_envs, projection_f, config)
    val_envs = StepSafeEnvironmentManager(_val_envs, projection_f, val_config)
    return envs, val_envs
