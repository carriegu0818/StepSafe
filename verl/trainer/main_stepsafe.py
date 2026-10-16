# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Entry point for training StepSafe with ASPO (SAPO decision credit + AFSD self-distillation).

    python -m verl.trainer.main_stepsafe [hydra overrides]

The configuration is ``verl/trainer/config/stepsafe_trainer.yaml``; ``scripts/train_stepsafe_4b.sh``
supplies the data, model and output paths.
"""

import os

import hydra
import ray
from omegaconf import OmegaConf, open_dict


@hydra.main(config_path="config", config_name="stepsafe_trainer", version_base=None)
def main(config):
    run_stepsafe(config)


def run_stepsafe(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        ray_init_kwargs = config.get("ray_init", {})
        runtime_env = OmegaConf.merge(get_ppo_ray_runtime_env(), ray_init_kwargs.get("runtime_env", {}))
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    from verl.utils.dataset.rl_dataset import RLHFDataset

    return RLHFDataset(data_files=data_paths, tokenizer=tokenizer, processor=processor, config=data_config)


def create_rl_sampler(data_config, dataset):
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # A seeded sampler makes the task order reproducible and resumable.
    if data_config.shuffle:
        generator = torch.Generator()
        generator.manual_seed(data_config.get("seed", 1))
        return RandomSampler(data_source=dataset, generator=generator)
    return SequentialSampler(data_source=dataset)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # AFSD enters the actor loss as lambda * L_AFSD (dp_actor.update_policy).
        afsd = config.algorithm.afsd
        with open_dict(config):
            config.actor_rollout_ref.actor.afsd_coef = afsd.coef
            config.actor_rollout_ref.actor.afsd_gate_beta = afsd.gate_beta

        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.actor.strategy not in ("fsdp", "fsdp2"):
            raise NotImplementedError("StepSafe trains with FSDP")
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRollout: global_pool_id}
        if config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # The replay outcome is logged as the episode return; SAPO does not use it for credit.
        from agent_system.reward_manager import EpisodeRewardManager

        reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        assert config.actor_rollout_ref.rollout.n == 1, "The G replays per task come from env.rollout.n; keep rollout.n=1"

        from agent_system.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        teacher_context = None
        if not config.trainer.get("val_only", False):
            from verl.trainer.ppo.afsd_context import AFSDTeacherContext

            if not afsd.trajectory_annotations or not afsd.step_annotations:
                raise ValueError("set algorithm.afsd.trajectory_annotations and algorithm.afsd.step_annotations")
            rubric = afsd.false_alarm_rubric
            # A fresh run starts with an empty false-alarm log and rubric; a resumed run keeps both.
            run_dir = config.trainer.default_local_dir
            has_checkpoint = os.path.exists(os.path.join(run_dir, "latest_checkpointed_iteration.txt")) or any(
                name.startswith("global_step_") for name in (os.listdir(run_dir) if os.path.isdir(run_dir) else []))
            if config.trainer.resume_mode == "disable" or not has_checkpoint:
                for path in (rubric.log_path, rubric.rubric_path):
                    if path and os.path.exists(path):
                        os.remove(path)
            teacher_context = AFSDTeacherContext(
                trajectory_annotations=afsd.trajectory_annotations,
                step_annotations=afsd.step_annotations,
                false_alarm_log=rubric.log_path,
                rubric_path=rubric.rubric_path,
                evolve_every=int(rubric.evolve_every),
                min_cluster=int(rubric.min_cluster),
                max_rules=int(rubric.max_rules),
                ucb_c=float(rubric.ucb_c),
                ask_risk_threshold=float(config.env.ask_risk_threshold),
            )
            print(f"[AFSD] lambda={afsd.coef} beta={afsd.gate_beta} rubric={rubric.rubric_path} "
                  f"log={rubric.log_path} evolve_every={rubric.evolve_every}")

        from verl.trainer.ppo.aspo_trainer import ASPORayTrainer

        trainer = ASPORayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            teacher_context=teacher_context,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
