#!/bin/bash
#SBATCH --job-name=dif-prcs-ins
#SBATCH --nodelist=server2
#SBATCH --output=slurm_logs/exp1%j.out
#SBATCH --error=slurm_logs/exp2%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=99:00:00

cd /home/kimhj/diffsion-processing-inpection-playground
source .venv/bin/activate

# uv run python test.py

# 실제 작업 실행
jupyter lab --ip=0.0.0.0 --port=9832 --no-browser
