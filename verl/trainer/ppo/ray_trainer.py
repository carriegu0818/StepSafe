# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO trainer infrastructure with a Ray single controller: worker groups,
dataloaders, validation on replayed trajectories, and checkpointing. The ASPO
training loop is in ``aspo_trainer.py``.
"""

import gzip
import json
import os
import shutil
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.monitor_metrics import (
    collect_terminal_records,
    episodes_from_validation,
    load_agentharm_labels,
    paper_table_metrics,
    validation_metrics,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

from agent_system.multi_turn_rollout import TrajectoryCollector

WorkerType = Type[Worker]


class Role(Enum):
    ActorRollout = 2
    RefPolicy = 4


@dataclass
class ResourcePoolManager:
    """Resource pool specification; pools are created before the worker groups."""

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count=1 merges all FSDP worker groups into one process per GPU.
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster")


def compute_response_mask(data: DataProto):
    """Attention mask of the response tokens."""
    response_length = data.batch["responses"].size(1)
    return data.batch["attention_mask"][:, -response_length:]


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """Runs on the driver; the actor, rollout and reference model live in Ray workers."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"
        assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0
        self.use_critic = False

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu is not None, "set actor.ppo_micro_batch_size_per_gpu"
            assert config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu is not None, "set rollout.log_prob_micro_batch_size_per_gpu"
            if self.use_reference_policy:
                assert config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu is not None, "set ref.log_prob_micro_batch_size_per_gpu"
        assert config.actor_rollout_ref.actor.loss_agg_mode == "token-mean", "SAPO weights assume token-mean aggregation"
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"
        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        val_batch_size = self.config.data.val_batch_size or len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps

        # Stop early without touching the LR schedule, which keeps the full-length
        # warmup of the training run. The stop step validates and saves like any last step.
        stop_after = self.config.trainer.get("stop_after_step", None)
        if stop_after:
            self.total_training_steps = min(self.total_training_steps, int(stop_after))
            print(f"Stopping after step {self.total_training_steps}; LR schedule still spans {total_training_steps} steps")

    # ── validation ───────────────────────────────────────────────────────────
    def _export_validation_records(self, records):
        """Write the replay records of this validation pass (latest + a gzipped copy per step)."""
        output_path = self.config.trainer.get("validation_records_path", None)
        if not output_path:
            return
        output_path = os.path.abspath(os.path.expanduser(str(output_path)))
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        temporary_path = f"{output_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            with open(temporary_path, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(temporary_path, output_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        step = int(getattr(self, "global_steps", 0) or 0)
        by_step = os.path.join(output_dir, "validation_records_by_step", f"validation_records_step{step:06d}.jsonl.gz")
        try:
            os.makedirs(os.path.dirname(by_step), exist_ok=True)
            with gzip.open(f"{by_step}.tmp", "wt", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(f"{by_step}.tmp", by_step)
        except Exception as error:  # noqa: BLE001 - a debug copy must never stop training
            print(f"[warn] could not keep per-step validation records at {by_step}: {error}")

    def _validate(self):
        """Replay every validation trajectory once and score the monitor."""
        terminal_records = []
        steps_by_traj = defaultdict(list)
        traj_records = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            for key in ("raw_prompt", "env_kwargs"):
                if key in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append(key)
            test_gen_batch = test_batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            test_output = self.traj_collector.multi_turn_loop(
                gen_batch=test_gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.val_envs,
                is_train=False,
            )
            traj_uids = test_output.non_tensor_batch["traj_uid"]
            for index, record in collect_terminal_records(test_output.non_tensor_batch.get("monitor_record"), traj_uids):
                terminal_records.append(record)
                traj_records.append((str(traj_uids[index]), record))
            verdict_steps = test_output.non_tensor_batch.get("verdict_step")
            if verdict_steps is not None:
                for uid, step in zip(traj_uids, verdict_steps):
                    if isinstance(step, dict) and step.get("act") is not None:
                        steps_by_traj[str(uid)].append(step)

        self._export_validation_records(terminal_records)
        metric_dict = validation_metrics(terminal_records)
        try:
            metric_dict.update(paper_table_metrics(
                episodes_from_validation(traj_records, steps_by_traj),
                load_agentharm_labels(self.config.trainer.get("agentharm_step_labels", None)),
            ))
        except Exception as error:  # noqa: BLE001 - a report metric must never stop training
            print(f"[warn] paper metrics not computed: {error!r}")
        # Read by _keep_top_k_checkpoints when the same step is saved.
        self._last_val_step_metrics = (self.global_steps, metric_dict)
        return metric_dict

    # ── workers ──────────────────────────────────────────────────────────────
    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.actor_rollout_ref, role="actor_rollout")
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()
        # Create the rollout last so vLLM can estimate the KV-cache memory better.
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    # ── checkpoints ──────────────────────────────────────────────────────────
    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        self.actor_rollout_wg.save_checkpoint(actor_local_path, None, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)

        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        self._keep_top_k_checkpoints()

    def _keep_top_k_checkpoints(self):
        """Keep the k saved steps with the highest validation metric, plus the latest.

        Off unless ``trainer.keep_top_k_ckpt`` is set. Each saved step's score is recorded in
        ``ckpt_val_metrics.json`` beside the checkpoints, so the ranking survives resumes. The
        latest step is always kept whole so training can resume; only scored steps are ever
        removed. With ``trainer.keep_top_k_prune_optim``, kept steps other than the latest drop
        their optimizer shards (their weights stay evaluable).
        """
        k = self.config.trainer.get("keep_top_k_ckpt", None)
        if not k:
            return
        key = self.config.trainer.get("keep_top_k_metric", "val/j")
        prune_optim = self.config.trainer.get("keep_top_k_prune_optim", False)
        root = self.config.trainer.default_local_dir
        scores_path = os.path.join(root, "ckpt_val_metrics.json")
        scores = {}
        if os.path.exists(scores_path):
            with open(scores_path) as f:
                scores = json.load(f).get("scores", {})
        step, metrics = getattr(self, "_last_val_step_metrics", (None, {}))
        if step == self.global_steps and key in metrics:
            scores[str(step)] = float(metrics[key])
            with open(f"{scores_path}.tmp", "w") as f:
                json.dump({"metric": key, "scores": scores}, f, indent=2, sort_keys=True)
            os.replace(f"{scores_path}.tmp", scores_path)
        else:
            print(f"[keep_top_k] step {self.global_steps} has no {key} from a validation pass at this step; it is kept unscored")

        saved = sorted(
            int(name.removeprefix("global_step_"))
            for name in os.listdir(root)
            if name.startswith("global_step_") and name.removeprefix("global_step_").isdigit()
        )
        ranked = sorted((s for s in saved if str(s) in scores), key=lambda s: (scores[str(s)], s), reverse=True)
        top = set(ranked[:k])
        for s in saved:
            if s >= self.global_steps or str(s) not in scores:
                continue
            folder = os.path.join(root, f"global_step_{s}")
            if s not in top:
                shutil.rmtree(folder, ignore_errors=True)
                print(f"[keep_top_k] removed {folder} ({key}={scores[str(s)]:.4f}, outside top {k})")
            elif prune_optim:
                actor_dir = os.path.join(folder, "actor")
                for name in os.listdir(actor_dir) if os.path.isdir(actor_dir) else []:
                    if name.startswith("optim_world_size_"):
                        os.remove(os.path.join(actor_dir, name))
        print(f"[keep_top_k] top {k} by {key}: " + ", ".join(f"{s}={scores[str(s)]:.4f}" for s in ranked[:k]))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}")
        self.actor_rollout_wg.load_checkpoint(os.path.join(global_step_folder, "actor"), del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            self.train_dataloader.load_state_dict(torch.load(dataloader_local_path, weights_only=False))
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder rows so each data-parallel rank gets a similar number of tokens."""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        metrics.update(log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix))
