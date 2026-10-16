"""Adaptive Feedback Self-Distillation (AFSD): loss and teacher batch, Section 2.4.

The monitor acts as its own teacher. The teacher sees the student's prompt with the
training-only context rho_t prepended (``afsd_context.py``) and scores the tokens the
student sampled; the student scores the same tokens without it. AFSD uses the gated
estimator of SDAR:

    Delta_{i,j} = stopgrad[log pi(o_ij | x_t, rho_t, o_i,<j)] - log pi(o_ij | x_t, o_i,<j)
    L_AFSD      = token-mean_{i,j}( stopgrad[sigmoid(beta * Delta_ij)] * Delta_ij )

It enters the objective as ``L_SAPO + lambda * L_AFSD`` (the ASPO objective); the decision
credit of SAPO is left unchanged. The loss covers every response token, including the
reasoning trace.
"""

from __future__ import annotations

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.model import compute_position_id_with_mask


def compute_afsd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gate_beta: float = 5.0,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """Gated self-distillation loss; gradients flow through the student only.

    Args:
        student_log_probs: (bs, response_length) log pi_theta(o_j | x, o_<j), with grad.
        teacher_log_probs: (bs, response_length) log pi_theta(o_j | x, rho, o_<j), frozen.
        response_mask: (bs, response_length) valid response tokens.
        gate_beta: beta of the sigmoid gate.
    """
    teacher_log_probs = teacher_log_probs.detach()
    delta = teacher_log_probs - student_log_probs.detach()
    gate = torch.sigmoid(gate_beta * delta).detach()
    gated = gate * (teacher_log_probs - student_log_probs)
    loss = agg_loss(loss_mat=gated, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        gate_mean = (gate * response_mask).sum() / mask_sum
        gate_active = ((gate > 0.5).float() * response_mask).sum() / mask_sum
        gap_mean = (delta * response_mask).sum() / mask_sum
    metrics = {
        "afsd/gate_mean": gate_mean.item(),
        "afsd/gate_active_ratio": gate_active.item(),
        "afsd/teacher_gap_mean": gap_mean.item(),
        "afsd/loss": loss.detach().item(),
    }
    return loss, metrics


def _head_tail_token_slice(token_ids: list[int], budget: int, tokenizer, head_fraction: float) -> list[int]:
    """Fit one prompt section while preserving both its opening and its end."""
    if len(token_ids) <= budget:
        return token_ids
    if budget <= 0:
        return []
    marker = tokenizer.encode("\n[older trajectory history omitted]\n", add_special_tokens=False)
    if budget <= len(marker) + 2:
        return token_ids[-budget:]
    content_budget = budget - len(marker)
    head = max(1, min(content_budget - 1, int(content_budget * head_fraction)))
    tail = content_budget - head
    return token_ids[:head] + marker + token_ids[-tail:]


def build_teacher_batch(
    batch: DataProto,
    teacher_context,
    tokenizer,
    max_prompt_length: int,
    enable_thinking: bool = True,
) -> DataProto:
    """Teacher inputs: [system turn with rho_t] + the student's prompt, same responses.

    ``teacher_context.context_for_sample`` renders rho_t for each sampled response; it
    also receives the decoded response so it can log false alarms.
    """
    bs = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)

    teacher_input_ids_list = []
    teacher_attention_mask_list = []
    teacher_position_ids_list = []

    for i in range(bs):
        original_input_ids = batch.batch["input_ids"][i]
        original_attention_mask = batch.batch["attention_mask"][i]
        prompt_length = original_input_ids.size(0) - response_length

        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]
        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        valid_start = valid_start[0].item() if len(valid_start) > 0 else 0
        prompt_text = tokenizer.decode(prompt_ids[valid_start:], skip_special_tokens=False)

        sample_metadata = {}
        for key, value in batch.non_tensor_batch.items():
            try:
                sample_metadata[key] = value[i]
            except Exception:  # noqa: BLE001
                sample_metadata[key] = value
        if getattr(teacher_context, "wants_student_response", False):
            response_mask = original_attention_mask[prompt_length:].bool()
            sample_metadata["student_response"] = tokenizer.decode(
                batch.batch["responses"][i][response_mask], skip_special_tokens=True)
        data_source = batch.non_tensor_batch.get("data_source", None)
        ds = "" if data_source is None else str(data_source[i])
        context_text = teacher_context.context_for_sample(prompt_text, sample_metadata, ds)

        messages = [{"role": "system", "content": f"[Privileged Information]\n{context_text}"}]
        try:
            context_prefix = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False, enable_thinking=enable_thinking)
        except TypeError:
            context_prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        context_ids = tokenizer.encode(context_prefix, add_special_tokens=False)
        student_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(context_ids) + len(student_ids) <= max_prompt_length:
            teacher_prompt_ids = context_ids + student_ids
        else:
            # Reserve most of the window for the student prompt: its head holds the
            # protocol and policy, its tail the current action.
            student_budget = min(len(student_ids), 3 * max_prompt_length // 4)
            context_budget = max_prompt_length - student_budget
            if len(context_ids) < context_budget:
                student_budget = min(len(student_ids), max_prompt_length - len(context_ids))
                context_budget = max_prompt_length - student_budget
            teacher_prompt_ids = _head_tail_token_slice(
                context_ids, context_budget, tokenizer, head_fraction=0.2,
            ) + _head_tail_token_slice(
                student_ids, student_budget, tokenizer, head_fraction=0.5,
            )

        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        actual_prompt_len = len(teacher_prompt_ids)
        pad_length = max_prompt_length - actual_prompt_len
        if pad_length > 0:  # left padding
            pad_ids = torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
            teacher_prompt_ids = torch.cat([pad_ids, teacher_prompt_ids])
            t_prompt_mask = torch.cat([
                torch.zeros(pad_length, dtype=torch.long),
                torch.ones(actual_prompt_len, dtype=torch.long),
            ])
        else:
            t_prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)

        response_ids = batch.batch["responses"][i]
        response_mask = original_attention_mask[-response_length:]
        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids])
        teacher_full_mask = torch.cat([t_prompt_mask, response_mask])
        teacher_position_ids = compute_position_id_with_mask(teacher_full_mask.unsqueeze(0))[0]

        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        teacher_position_ids_list.append(teacher_position_ids)

    return DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(teacher_input_ids_list),
            "attention_mask": torch.stack(teacher_attention_mask_list),
            "position_ids": torch.stack(teacher_position_ids_list),
            "responses": batch.batch["responses"],
        },
    )
