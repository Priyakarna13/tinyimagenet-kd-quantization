#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=tinyimg_teacher
#SBATCH --mem=64G
#SBATCH -c 12
#SBATCH -t 0-18:00:00
#SBATCH -G a100:1
#SBATCH -o /data/pjiang18/SLP/ModelCompression/logs/teacher_%j.out
#SBATCH -e /data/pjiang18/SLP/ModelCompression/logs/teacher_%j.err
#SBATCH --export=ALL

module load mamba/latest
source activate /data/pjiang18/SLP/envs/llm-bench

# System stability tweaks
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

ulimit -n 4096

cd /data/pjiang18/SLP/ModelCompression

python train_teacher_tinyimagenet.py \
  --epochs 200 \
  --patience 20 \
  --min_delta 0.001 \
  --batch_size 256 \
  --lr 0.001 \
  --weight_decay 1e-4 \
  --num_workers 8 \
  --seed 42