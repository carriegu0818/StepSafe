#!/usr/bin/env bash
# Evaluate a StepSafe checkpoint on the Table 1 trajectories (4,033 rows: ASSE-Safety,
# ASSE-Security, R-Judge, AgentHarm, ATBench) with the validation protocol of training:
# greedy decoding, thinking on, 2,048 response tokens, full visible history (12,000 characters).
# Append env.validation.action_history_mode=evidence_retrieval for evidence retrieval over
# earlier records (memory 4,096 / read 2,048 tokens). The prompt limit is raised to 20,480 tokens:
# with the full history, a few ATBench trajectories (long tool definitions) exceed the 16,384
# tokens of training.
# Prints and logs paper/<dataset>/{acc,mf1,util,j,asr} and paper/Overall/{bir,ble,stop}.
#
# A released (Hugging Face) checkpoint:
#   MODEL=/path/to/StepSafe-4B bash scripts/eval_stepsafe.sh
# A kept step of a training run (FSDP checkpoint; MODEL is the run's warm-start checkpoint):
#   MODEL=checkpoints/StepSafe-4B-SFT CKPT=outputs/stepsafe_4b_s42/global_step_320 \
#       bash scripts/eval_stepsafe.sh
#   (outputs/<run>/ckpt_val_metrics.json lists the kept steps and their validation J.)
#
# Optional: STEPSAFE_DATA [data], OUT_DIR [outputs/eval/<name>], N_GPUS [2]. Extra Hydra overrides can be appended.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/harness_audit${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

STEPSAFE_DATA="${STEPSAFE_DATA:-${REPO_ROOT}/data}"
: "${MODEL:?set MODEL to a Hugging Face checkpoint directory}"
# Hydra may change the working directory, so every path is made absolute.
STEPSAFE_DATA="$(realpath -m "${STEPSAFE_DATA}")"; MODEL="$(realpath -m "${MODEL}")"
[ -z "${VAL_FILE:-}" ] || VAL_FILE="$(realpath -m "${VAL_FILE}")"
VAL_FILE="${VAL_FILE:-${STEPSAFE_DATA}/eval/table1.parquet}"
AGENTHARM_STEP_LABELS="${STEPSAFE_DATA}/eval/agentharm_step_labels.jsonl"
for required in "${VAL_FILE}" "${AGENTHARM_STEP_LABELS}" "${MODEL}/config.json"; do
    [ -f "${required}" ] || { echo "ERROR: missing ${required} (python scripts/build_data.py --only eval)" >&2; exit 1; }
done

if [ -n "${CKPT:-}" ]; then
    [ -d "${CKPT}/actor" ] || { echo "ERROR: ${CKPT} is not a training checkpoint (no actor/)" >&2; exit 1; }
    RESUME=(trainer.resume_mode=resume_path trainer.resume_from_path="$(cd "${CKPT}" && pwd)")
    NAME="$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}")"
else
    RESUME=(trainer.resume_mode=disable)
    NAME="$(basename "${MODEL}")"
fi
OUT_DIR="$(realpath -m "${OUT_DIR:-${REPO_ROOT}/outputs/eval/${NAME}}")"
mkdir -p "${OUT_DIR}"
LOGGER="['console','wandb']"
[ -n "${WANDB_API_KEY:-}" ] || LOGGER="['console']"

python -m verl.trainer.main_stepsafe \
    data.train_files="['${VAL_FILE}']" \
    data.val_files="['${VAL_FILE}']" \
    actor_rollout_ref.model.path="${MODEL}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="${N_GPUS:-2}" \
    trainer.project_name="${WANDB_PROJECT:-stepsafe}" \
    trainer.experiment_name="eval_${NAME}" \
    trainer.logger="${LOGGER}" \
    trainer.default_local_dir="${OUT_DIR}" \
    trainer.agentharm_step_labels="${AGENTHARM_STEP_LABELS}" \
    data.max_prompt_length=20480 \
    actor_rollout_ref.rollout.max_model_len=22528 \
    actor_rollout_ref.rollout.max_num_batched_tokens=22528 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=22528 \
    "${RESUME[@]}" \
    "$@" 2>&1 | tee "${OUT_DIR}/eval.log"
echo "[eval] replay records: ${OUT_DIR}/validation/validation_records.jsonl"
