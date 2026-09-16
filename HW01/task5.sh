#!/bin/bash
#SBATCH --partition=instruction
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=CheckSlurmEnv
#SBATCH --output=task5-%j.out
#SBATCH --error=task5-%j.err

set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    printf 'Please submit this script with sbatch on Euler.\n' >&2
    exit 1
fi

# Q5(a): compare the actual working directory with the submission directory.
printf 'Working directory (pwd): %s\n' "$(pwd -P)"
printf 'SLURM_SUBMIT_DIR: %s\n' "$SLURM_SUBMIT_DIR"

# Q5(b): compare this ID with the number reported by sbatch.
printf 'SLURM_JOB_ID: %s\n' "$SLURM_JOB_ID"
printf 'SLURM_JOB_NAME: %s\n' "$SLURM_JOB_NAME"
printf 'Compute node: %s\n' "$(hostname)"
