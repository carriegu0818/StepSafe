#!/usr/bin/env bash
# Train StepSafe-4B with ASPO (SAPO decision credit + AFSD self-distillation), starting from
# the supervised warm start. All hyperparameters live in verl/trainer/config/stepsafe_trainer.yaml;
# this script supplies paths and checks the inputs.
#
#   python scripts/build_data.py            # once: builds data/rl/ and data/eval/ (README.md, "Data")
#   SFT_CKPT=checkpoints/StepSafe-4B-SFT bash scripts/train_stepsafe_4b.sh
#
# Required:
#   SFT_CKPT        directory of the warm-start checkpoint, StepSafe-4B-SFT (README.md, "Reproducing")
# Optional:
#   STEPSAFE_DATA   data directory with rl/ and eval/ [data]
#   RUN_DIR         run directory (checkpoints, false-alarm rubric, validation records)
#                   [outputs/stepsafe_4b_s${SEED}]
#   SEED            [42]
#   STOP_AFTER_STEP [500]
#   N_GPUS          [2]  (the paper used 2x B200; 2x 96 GB GPUs also fit)
#   WANDB_PROJECT   [stepsafe]; W&B logging is on when WANDB_API_KEY is set.
# Extra Hydra overrides can be appended as arguments.
#
# Resubmitting with the same RUN_DIR resumes from its latest checkpoint (trainer.resume_mode=auto).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/harness_audit${PYTHONPATH:+:${PYTHONPATH}}"

STEPSAFE_DATA="${STEPSAFE_DATA:-${REPO_ROOT}/data}"
: "${SFT_CKPT:?set SFT_CKPT to the warm-start checkpoint (Hugging Face format)}"
SEED="${SEED:-42}"
STOP_AFTER_STEP="${STOP_AFTER_STEP:-500}"
N_GPUS="${N_GPUS:-2}"
RUN_NAME="${RUN_NAME:-stepsafe_4b_s${SEED}}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/${RUN_NAME}}"
# Hydra may change the working directory, so every path is made absolute.
STEPSAFE_DATA="$(realpath -m "${STEPSAFE_DATA}")"; SFT_CKPT="$(realpath -m "${SFT_CKPT}")"; RUN_DIR="$(realpath -m "${RUN_DIR}")"

TRAIN_FILE="${STEPSAFE_DATA}/rl/train.parquet"
TRAJECTORY_ANNOTATIONS="${STEPSAFE_DATA}/rl/trajectory_annotations.jsonl"
STEP_ANNOTATIONS="${STEPSAFE_DATA}/rl/step_annotations.jsonl"
# Validation every 10 steps: the ASSE-Safety, ASSE-Security, R-Judge, AgentHarm (100 each) and
# ATBench (100) episodes of the 750-episode validation set. VAL_FILES overrides (comma-separated).
VAL_FILES="${VAL_FILES:-${STEPSAFE_DATA}/eval/validation/assebench.parquet,${STEPSAFE_DATA}/eval/validation/atbench.parquet}"
VAL_LIST=""
IFS=',' read -r -a _val_files <<< "${VAL_FILES}"
for f in "${_val_files[@]}"; do
    f="$(realpath -m "${f}")"
    [ -f "${f}" ] || { echo "ERROR: missing ${f}; build it with: python scripts/build_data.py" >&2; exit 1; }
    VAL_LIST="${VAL_LIST:+${VAL_LIST},}'${f}'"
done
AGENTHARM_STEP_LABELS="${STEPSAFE_DATA}/eval/agentharm_step_labels.jsonl"
[ -f "${TRAIN_FILE}" ] || { echo "ERROR: missing ${TRAIN_FILE}; build it with: python scripts/build_data.py" >&2; exit 1; }
for required in "${TRAIN_FILE}" "${TRAJECTORY_ANNOTATIONS}" "${STEP_ANNOTATIONS}" \
                "${AGENTHARM_STEP_LABELS}" "${SFT_CKPT}/config.json"; do
    [ -f "${required}" ] || { echo "ERROR: missing ${required}" >&2; exit 1; }
done

# The paper's runs used the Triton attention backend in vLLM (FlashAttention corrupted long
# decodes on sm_120 GPUs); keep it for exact reproduction.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

LOGGER="['console','wandb']"
[ -n "${WANDB_API_KEY:-}" ] || LOGGER="['console']"

# Release check: prompts, annotations and dataset sizes (CPU, a few seconds).
python scripts/check_data.py --data "${STEPSAFE_DATA}"

mkdir -p "${RUN_DIR}"
latest=$(cat "${RUN_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [ "${latest}" -ge "${STOP_AFTER_STEP}" ]; then
    echo "[stepsafe] ${RUN_DIR} is at step ${latest} (stop ${STOP_AFTER_STEP}); nothing to train"
    exit 0
fi
# A resumed run starts at a step that was validated when it was saved (test_freq = save_freq),
# so it skips the pre-training validation; a fresh run validates the warm start at step 0.
VAL_BEFORE_TRAIN=True
[ "${latest}" -gt 0 ] && VAL_BEFORE_TRAIN=False
echo "[stepsafe] run=${RUN_DIR} step ${latest} -> ${STOP_AFTER_STEP}, init=${SFT_CKPT}, seed=${SEED}"
exec python -m verl.trainer.main_stepsafe \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="[${VAL_LIST}]" \
    data.seed="${SEED}" \
    actor_rollout_ref.model.path="${SFT_CKPT}" \
    algorithm.afsd.trajectory_annotations="${TRAJECTORY_ANNOTATIONS}" \
    algorithm.afsd.step_annotations="${STEP_ANNOTATIONS}" \
    env.seed="${SEED}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.stop_after_step="${STOP_AFTER_STEP}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.project_name="${WANDB_PROJECT:-stepsafe}" \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.logger="${LOGGER}" \
    trainer.default_local_dir="${RUN_DIR}" \
    trainer.agentharm_step_labels="${AGENTHARM_STEP_LABELS}" \
    "$@"
