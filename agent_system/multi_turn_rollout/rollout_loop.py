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

"""Replay rollouts: the monitor steps through recorded trajectories in parallel.

Each training task is replayed ``env.rollout.n`` (G) times. At every step the
monitor's prompt is rebuilt from the replay environment, the policy samples one
verdict, and the environment advances or ends the run. Every visited step becomes
one training row that carries, besides the tokens:

* ``uid`` (task) and ``traj_uid`` (run), and ``anchor_obs`` (the visible prefix);
* ``stage_context`` (landmarks and sampled action, for the SAPO stage score);
* ``intervention_target`` (annotated class d*_t) and ``intervention_predicted`` (D(v));
* ``verdict_step`` and, on the last step of a run, ``monitor_record`` (validation metrics).
"""

import uuid
from typing import Dict, List

import numpy as np
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask

from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(self, item: int, gen_batch: DataProto, obs: Dict):
        """Tokenize the monitor prompt of one environment: a single user turn."""
        data_source = gen_batch.non_tensor_batch["data_source"][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        obs_texts = obs.get("text", None)
        obs_anchors = obs.get("anchor", None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        if obs_text is None:
            print("Warning: No text observation found!")
            obs_text = ""

        chat = np.array([{"content": obs_text, "role": "user"}])
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False, **apply_chat_template_kwargs,
        )
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        raw_prompt_ids = self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        row_dict = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "anchor_obs": obs_anchor,
            "index": item,
            "data_source": data_source,
        }
        if self.config.data.get("return_raw_chat", False):
            row_dict["raw_prompt"] = chat.tolist()
        return row_dict

    def preprocess_batch(self, gen_batch: DataProto, obs: Dict) -> DataProto:
        batch_size = len(gen_batch.batch["input_ids"])
        processed_samples = [self.preprocess_single_sample(item=item, gen_batch=gen_batch, obs=obs) for item in range(batch_size)]
        return DataProto.from_single_dict(data=collate_fn(processed_samples), meta_info=gen_batch.meta_info)

    def gather_rollout_data(
        self,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
    ) -> DataProto:
        """One row per step that a run actually visited."""
        batch_size = len(total_batch_list)
        success_rate = {key: np.mean(value) for key, value in success.items()}
        effective_batch = []
        for bs in range(batch_size):
            for step_idx, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data["traj_uid"], "data is not from the same trajectory"
                if data["active_masks"]:
                    data["episode_rewards"] = episode_rewards[bs]
                    data["episode_lengths"] = episode_lengths[bs]
                    data["tool_callings"] = tool_callings[bs]
                    data["turn_step"] = step_idx
                    for key, value in success_rate.items():
                        data[key] = value
                    effective_batch.append(data)
        # The replay record exists only on a run's last step; the collator needs every key on every row.
        for key in ("monitor_record", "verdict_step"):
            if any(key in data for data in effective_batch):
                for data in effective_batch:
                    data.setdefault(key, None)
        return DataProto.from_single_dict(data=collate_fn(effective_batch))

    def vanilla_multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg, envs: EnvironmentManagerBase):
        batch_size = len(gen_batch.batch)

        # Keep the row-level data_source inside env_kwargs so replay records can be grouped.
        env_kwargs = gen_batch.non_tensor_batch.pop("env_kwargs", None)
        data_sources = gen_batch.non_tensor_batch.get("data_source")
        if env_kwargs is not None and data_sources is not None:
            patched_kwargs = []
            for kw, data_source in zip(list(env_kwargs), list(data_sources)):
                kw = dict(kw or {})
                kw.setdefault("data_source", str(data_source))
                patched_kwargs.append(kw)
            env_kwargs = patched_kwargs
        obs, infos = envs.reset(kwargs=env_kwargs)
        assert len(gen_batch.batch) == len(obs["text"]), f"gen_batch size {len(gen_batch.batch)} does not match obs size {len(obs['text'])}"

        # All G replays of a task share one uid.
        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(batch_size)], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            batch_input = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            batch_input.meta_info = gen_batch.meta_info

            # Pad to a multiple of the data-parallel size.
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch["uid"] = uid_batch
            batch.non_tensor_batch["traj_uid"] = traj_uid
            batch = batch.union(batch_output)

            text_actions = self.tokenizer.batch_decode(
                batch.batch["responses"],
                skip_special_tokens=not getattr(envs, "preserve_special_tokens_for_parser", False),
            )
            next_obs, rewards, dones, infos = envs.step(text_actions)
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            if "is_action_valid" in infos[0]:
                batch.non_tensor_batch["is_action_valid"] = np.array([info["is_action_valid"] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch["is_action_valid"] = np.ones(batch_size, dtype=bool)
            # Training-only supervision of each decision (finished runs report defaults).
            for key in ("intervention_target", "intervention_predicted"):
                if any(key in info for info in infos):
                    batch.non_tensor_batch[key] = np.array([info.get(key, False) for info in infos], dtype=bool)
            if any("stage_context" in info for info in infos):
                batch.non_tensor_batch["stage_context"] = np.array([info.get("stage_context") for info in infos], dtype=object)
            # Per-step verdicts, for scoring validation with the paper's rules.
            if any("parsed_act" in info for info in infos):
                batch.non_tensor_batch["verdict_step"] = np.array(
                    [{"step": _step, "cursor": info.get("cursor"), "act": info.get("parsed_act"),
                      "risk": info.get("parsed_risk")} for info in infos],
                    dtype=object,
                )
            if any("monitor_record" in info for info in infos):
                batch.non_tensor_batch["monitor_record"] = np.array([info.get("monitor_record") for info in infos], dtype=object)

            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1
            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch["active_masks"] = torch_to_numpy(active_masks, is_object=True)

            batch_list: list[dict] = to_list_of_dict(batch)
            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            is_done = np.logical_or(is_done, dones)
            obs = next_obs
            if is_done.all():
                break

        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

    def multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg, envs: EnvironmentManagerBase, is_train: bool = True) -> DataProto:
        """Replay a batch of tasks (G times each when training) and collect the step rows."""
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings = \
            self.vanilla_multi_turn_loop(gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, envs=envs)
        assert len(total_batch_list) == len(total_episode_rewards) == len(total_episode_lengths) == len(total_traj_uid)
        return self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=total_tool_callings,
        )
