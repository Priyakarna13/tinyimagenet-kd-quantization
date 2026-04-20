#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=tinyimg_student_kd
#SBATCH --mem=64G
#SBATCH -c 12
#SBATCH -t 0-18:00:00
#SBATCH -G a100:1
#SBATCH -o /data/pjiang18/SLP/ModelCompression/logs/student_kd_%j.out
#SBATCH -e /data/pjiang18/SLP/ModelCompression/logs/student_kd_%j.err
#SBATCH --export=NONE

module load mamba/latest
source activate /data/pjiang18/SLP/envs/llm-bench

cd /data/pjiang18/SLP/ModelCompression

python train_student_kd_tinyimagenet.py \
  --epochs 100 \
  --patience 15 \
  --min_delta 0.001 \
  --batch_size 256 \
  --lr 0.001 \
  --weight_decay 1e-4 \
  --num_workers 8 \
  --seed 42 \
  --temperature 4.0 \
  --alpha 0.7 \
  --feature_beta 2.0 \
  --teacher_ckpt /data/pjiang18/SLP/ModelCompression/checkpoints/tinyimagenet_resnet50_teacher_best.pth \
  --teacher_results /data/pjiang18/SLP/ModelCompression/results/tinyimagenet_teacher_results.json \
  --student_ckpt /data/pjiang18/SLP/ModelCompression/checkpoints/tinyimagenet_resnet18_student_best.pth \
  --results_file /data/pjiang18/SLP/ModelCompression/results/tinyimagenet_kd_student_results.json