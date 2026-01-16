#!/bin/bash
#SBATCH --job-name=dffs-prs-inspection
#SBATCH --nodelist=ubuntu
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=99:00:00

# -----------------------------
# Resolve directory of this script (works regardless of where sbatch is called)
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"

# -----------------------------
# Ensure slurm log directory exists (relative to script location)
# -----------------------------
mkdir -p "${SCRIPT_DIR}/slurm_logs"

# -----------------------------
# Move to project root (script dir)
# -----------------------------
cd "${SCRIPT_DIR}" || exit 1

# -----------------------------
# Activate virtual environment
# -----------------------------
source .venv/bin/activate

# -----------------------------
# Run job
# -----------------------------
jupyter lab --ip=0.0.0.0 --port=9832 --no-browser
