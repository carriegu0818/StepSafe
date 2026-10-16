#!/bin/bash
# Slurm template: evaluate a StepSafe checkpoint on the Table 1 trajectories (2 GPUs, ~1 hour).
#
#   MODEL=/path/to/StepSafe-4B sbatch scripts/slurm/eval_stepsafe.sbatch
#   MODEL=checkpoints/StepSafe-4B-SFT CKPT=outputs/stepsafe_4b_s42/global_step_320 \
#       sbatch scripts/slurm/eval_stepsafe.sbatch
#SBATCH --job-name=stepsafe-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --partition=<gpu partition>
##SBATCH --account=<account>
set -euo pipefail

# --- environment (edit) -------------------------------------------------------------
# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate stepsafe
# ------------------------------------------------------------------------------------

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
export RAY_TMPDIR="${TMPDIR}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
bash scripts/eval_stepsafe.sh "$@"
