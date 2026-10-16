#!/usr/bin/env bash
# Submit a Slurm script as N chained jobs; each part resumes the run from its latest checkpoint.
#
#   SFT_CKPT=... bash scripts/slurm/submit_chain.sh scripts/slurm/train_stepsafe_4b.sbatch 6
#
# Extra sbatch options can follow the part count, e.g. --partition=gpu.
set -euo pipefail
script="${1:?usage: submit_chain.sh <sbatch script> [parts] [sbatch options...]}"
parts="${2:-6}"
shift $(( $# >= 2 ? 2 : 1 ))
previous=""
for part in $(seq 1 "${parts}"); do
    dependency=()
    [ -n "${previous}" ] && dependency=(--dependency="afterany:${previous}")
    previous=$(sbatch --parsable --export=ALL "${dependency[@]}" "$@" "${script}")
    echo "part ${part}/${parts}: job ${previous}"
done
