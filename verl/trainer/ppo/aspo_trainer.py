"""Adaptive Stage-Aligned Policy Optimization (ASPO): the StepSafe training loop.

One update (Algorithm 1 of the paper):

1. Replay each training task G times with the current policy (``TrajectoryCollector``);
   every visited step yields one response row.
2. Render the teacher context rho_t for each response (``afsd_context.py``) and score the
   student's sampled tokens with and without it (teacher and student share parameters).
3. Group responses by visible prefix and compute the SAPO advantages and class-balanced
   weights (``sapo.py``).
4. Update the policy with L_SAPO + lambda * L_AFSD (+ the KL loss to the reference policy);
   the AFSD term is computed inside the actor (``verl/workers/actor/dp_actor.py``).
5. Every ``evolve_every`` teacher calls, mine false-alarm rules with the live policy.
"""

from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.afsd import build_teacher_batch
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _timer, compute_response_mask
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.sapo import compute_sapo_advantage
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import masked_mean

from agent_system.multi_turn_rollout import adjust_batch

# Training metrics the paper reports or plots. Validation logs val/* and paper/* (monitor_metrics.py).
# trainer.log_all_metrics=True logs every metric instead (rollout, SAPO and AFSD diagnostics, timing).
PAPER_TRAINING_METRICS = (
    "actor/pg_loss", "actor/kl_loss", "afsd/loss", "actor/entropy_loss",
    "actor/grad_norm", "actor/lr", "response_length/mean", "timing_s/step",
)


class ASPORayTrainer(RayPPOTrainer):
    def __init__(self, *args, teacher_context=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_context = teacher_context

    def _loggable(self, metrics: dict) -> dict:
        if self.config.trainer.get("log_all_metrics", False):
            return metrics
        return {k: v for k, v in metrics.items() if k in PAPER_TRAINING_METRICS or k.startswith(("val/", "paper/"))}

    # ── AFSD teacher ─────────────────────────────────────────────────────────
    def _teacher_enable_thinking(self) -> bool:
        """The teacher scores the student's tokens under the student's chat template."""
        value = self.config.data.get("apply_chat_template_kwargs", {}).get("enable_thinking", False)
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        """log pi_theta(o_j | x, rho_t, o_<j) of the student's own tokens."""
        teacher_batch = build_teacher_batch(
            batch=batch,
            teacher_context=self.teacher_context,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            enable_thinking=self._teacher_enable_thinking(),
        )
        return self.actor_rollout_wg.compute_log_prob(teacher_batch).batch["old_log_probs"]

    def _install_rule_writer(self) -> None:
        """The live policy writes and judges candidate false-alarm rules."""
        if self.teacher_context is not None and hasattr(self.teacher_context, "set_generate_fn"):
            self.teacher_context.set_generate_fn(self._generate_with_policy)
            print("[AFSD] false-alarm rules are written and judged by the current policy")

    def _generate_with_policy(self, system: str, user: str) -> str:
        """Greedy-decode one completion from the live rollout engine."""
        tok = self.tokenizer
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            prompt_text = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=self._teacher_enable_thinking())
        except TypeError:
            prompt_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tok.encode(prompt_text, add_special_tokens=False)
        max_prompt_length = self.config.data.max_prompt_length
        if len(input_ids) > max_prompt_length:
            input_ids = input_ids[-max_prompt_length:]
        seq_len = len(input_ids)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        pad_len = max_prompt_length - seq_len
        ids = torch.tensor([pad_id] * pad_len + input_ids, dtype=torch.long).unsqueeze(0)
        attn = torch.tensor([0] * pad_len + [1] * seq_len, dtype=torch.long).unsqueeze(0)
        pos = compute_position_id_with_mask(attn)
        gen_batch = DataProto.from_dict(tensors={"input_ids": ids, "attention_mask": attn, "position_ids": pos})
        gen_batch.meta_info = {"do_sample": False, "validate": False, "recompute_log_prob": False}
        padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        out = unpad_dataproto(self.actor_rollout_wg.generate_sequences(padded), pad_size=pad_size)
        return tok.decode(out.batch["responses"][0], skip_special_tokens=True)

    # ── training loop ────────────────────────────────────────────────────────
    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self._install_rule_writer()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=self._loggable(val_metrics), step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training")
        self.global_steps += 1
        last_val_metrics = None
        afsd_coef = self.config.algorithm.afsd.coef

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                for key in ("raw_prompt", "env_kwargs"):
                    if key in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append(key)
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # 1. Replay every task G times; one row per visited step.
                    with _timer("gen", timing_raw):
                        batch = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )
                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # Replay outcome (episode return): logged, not used for credit.
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=batch.batch["response_mask"],
                                                loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode)
                        metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # 2. AFSD teacher pass under rho_t.
                    with _timer("teacher_forward", timing_raw):
                        batch.batch["teacher_log_probs"] = self._compute_teacher_log_probs(batch)

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # 3. SAPO decision credit.
                    with _timer("adv", timing_raw):
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        advantages, returns, sapo_stats = compute_sapo_advantage(
                            batch, batch.batch["response_mask"], balance=self.config.algorithm.sapo.get("balance", None))
                        batch.batch["advantages"] = advantages
                        batch.batch["returns"] = returns
                        metrics.update({f"sapo/{k}": v for k, v in sapo_stats.items()})

                        response_mask = batch.batch["response_mask"]
                        delta_t = (batch.batch["teacher_log_probs"] - batch.batch["old_log_probs"]) * response_mask
                        metrics["afsd/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                        metrics["afsd/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()
                        metrics["afsd/coef"] = afsd_coef

                    # 4. Policy update: L_SAPO + lambda * L_AFSD (+ KL).
                    with _timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = False
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw,
                                                          n_gpus=self.resource_pool_manager.get_n_gpus()))
                logger.log(data=self._loggable(metrics), step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
