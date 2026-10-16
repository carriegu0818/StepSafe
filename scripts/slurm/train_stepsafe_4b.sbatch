#!/bin/bash
# Slurm template: StepSafe-4B, the full model of the paper (Qwen3-4B, warm start + ASPO).
# Adjust the #SBATCH lines and the environment block to your cluster, then
#
#   SFT_CKPT=checkpoints/StepSafe-4B-SFT sbatch scripts/slurm/train_stepsafe_4b.sbatch
#
# Cost: ~5-6 min per training step on 2x B200 plus a few minutes of validation (500 episodes)
# every 10 steps, i.e. about two days for 500 steps. Runs resume from their latest checkpoint, so a
# run longer than the partition's time limit can be split into chained jobs
# (scripts/slurm/submit_chain.sh). A finished run keeps about 100 GB of checkpoints.
#SBATCH --job-name=stepsafe-4b
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --partition=<gpu partition>
##SBATCH --account=<account>
set -euo pipefail

# --- environment (edit) -------------------------------------------------------------
# Activate the Python environment of README.md ("Installation"), e.g.
#   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate stepsafe
# Keep caches off a small home quota if needed:
#   export HF_HOME=/path/to/cache/huggingface
# W&B logging is enabled when WANDB_API_KEY is set (export it or keep it in your env).
# ------------------------------------------------------------------------------------

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

# Ray and vLLM temporary files go to the job's local scratch when available.
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
export RAY_TMPDIR="${TMPDIR}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"

export SEED="${SEED:-42}"
export STOP_AFTER_STEP="${STOP_AFTER_STEP:-500}"
export RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/stepsafe_4b_s${SEED}}"
bash scripts/train_stepsafe_4b.sh "$@"
